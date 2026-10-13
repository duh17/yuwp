import Foundation
@preconcurrency import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// Qwen3.5-MoE text backbone of Drama 3: 3 gated-delta (linear attention) layers per gated
// full-attention layer, every layer a 256-expert top-8 MoE with a gated shared expert.
// Converted weights already carry +1-shifted RMSNorm gammas.

final class Drama3LinearAttentionState {
    var conv: MLXArray?
    var recurrent: MLXArray?
}

final class Drama3BackboneCache {
    var linear: [Drama3LinearAttentionState?]
    var full: [KVCacheSimple?]
    var offset = 0

    init(config: Drama3Config.Backbone) {
        linear = (0 ..< config.layerCount).map { config.isLinearLayer($0) ? Drama3LinearAttentionState() : nil }
        full = (0 ..< config.layerCount).map { config.isLinearLayer($0) ? nil : KVCacheSimple() }
    }
}

final class Drama3GatedRMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return (silu(gate.asType(.float32)) * normed.asType(.float32)).asType(x.dtype)
    }
}

enum Drama3GatedDelta {
    static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "drama3_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
            }
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);
              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
            }
            """
    )

    /// Gated delta rule over T steps; the recurrent state stays float32.
    static func update(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)
        let outputs = kernel(
            [q, k, v, g, beta, state, MLXArray(T)],
            template: [
                ("InT", q.dtype), ("StT", state.dtype),
                ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape],
            outputDTypes: [q.dtype, state.dtype]
        )
        return (outputs[0], outputs[1])
    }
}

final class Drama3GatedDeltaNet: Module {
    let valueHeads: Int
    let keyHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let keyDim: Int
    let convKernel: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Drama3GatedRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Drama3Config.Backbone) {
        valueHeads = config.linearValueHeads
        keyHeads = config.linearKeyHeads
        keyHeadDim = config.linearKeyHeadDim
        valueHeadDim = config.linearValueHeadDim
        keyDim = keyHeads * keyHeadDim
        convKernel = config.linearConvKernel
        let valueDim = valueHeads * valueHeadDim
        convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim, kernelSize: convKernel,
            groups: convDim, bias: false
        )
        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([valueHeads])
        _aLog.wrappedValue = MLXArray.zeros([valueHeads])
        _norm.wrappedValue = Drama3GatedRMSNorm(dimensions: valueHeadDim, eps: config.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, state: Drama3LinearAttentionState) -> MLXArray {
        let B = x.dim(0)
        let S = x.dim(1)
        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, valueHeads, valueHeadDim)
        let b = inProjB(x)
        let a = inProjA(x)

        let convState = state.conv ?? MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        state.conv = convInput[0..., (convInput.dim(1) - (convKernel - 1))...]
        let convOut = silu(conv1d(convInput))

        let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let invScale = 1 / Float(keyHeadDim).squareRoot()
        let rmsEps: Float = 1e-6 * invScale * invScale
        let q = MLXArray(invScale * invScale).asType(x.dtype)
            * MLXFast.rmsNorm(parts[0].reshaped(B, S, keyHeads, keyHeadDim), weight: MLXArray.mlxNone, eps: rmsEps)
        let k = MLXArray(invScale).asType(x.dtype)
            * MLXFast.rmsNorm(parts[1].reshaped(B, S, keyHeads, keyHeadDim), weight: MLXArray.mlxNone, eps: rmsEps)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)

        let beta = sigmoid(b)
        let g = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
        let recurrent = state.recurrent
            ?? MLXArray.zeros([B, valueHeads, valueHeadDim, keyHeadDim], dtype: .float32)
        let (out, newState) = Drama3GatedDelta.update(q: q, k: k, v: v, g: g, beta: beta, state: recurrent)
        state.recurrent = newState
        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }
}

final class Drama3FullAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let rotaryDim: Int
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: Drama3Config.Backbone) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        rotaryDim = config.rotaryDim
        ropeTheta = config.ropeTheta
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCacheSimple) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let queryAndGate = qProj(x).reshaped(B, L, heads, 2 * headDim)
        let halves = split(queryAndGate, parts: 2, axis: -1)
        let gate = halves[1].reshaped(B, L, heads * headDim)
        var queries = qNorm(halves[0]).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache.offset
        queries = MLXFast.RoPE(queries, dimensions: rotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        keys = MLXFast.RoPE(keys, dimensions: rotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        let (allKeys, allValues) = cache.update(keys: keys, values: values)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues,
            scale: 1 / Float(headDim).squareRoot(), mask: L > 1 ? .causal : .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, heads * headDim)
        return oProj(output * sigmoid(gate))
    }
}

final class Drama3DenseMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class Drama3SparseMoE: Module {
    let topK: Int
    let normalizeTopK: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Drama3DenseMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ config: Drama3Config.Backbone) {
        topK = config.expertsPerToken
        normalizeTopK = config.normTopKProbability
        _gate.wrappedValue = Linear(config.hiddenSize, config.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize, numExperts: config.expertCount
        )
        _sharedExpert.wrappedValue = Drama3DenseMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let probabilities = softmax(gate(x), axis: -1, precise: true)
        let kth = probabilities.dim(-1) - topK
        let indices = argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = takeAlong(probabilities, indices, axis: -1)
        if normalizeTopK {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        let routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

final class Drama3BackboneLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "linear_attn") var linearAttention: Drama3GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var fullAttention: Drama3FullAttention?
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Drama3SparseMoE

    init(_ config: Drama3Config.Backbone, index: Int) {
        isLinear = config.isLinearLayer(index)
        if isLinear {
            _linearAttention.wrappedValue = Drama3GatedDeltaNet(config)
        } else {
            _fullAttention.wrappedValue = Drama3FullAttention(config)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = Drama3SparseMoE(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Drama3BackboneCache, index: Int) -> MLXArray {
        let mixed: MLXArray
        if let linearAttention, let state = cache.linear[index] {
            mixed = linearAttention(inputNorm(x), state: state)
        } else if let fullAttention, let kv = cache.full[index] {
            mixed = fullAttention(inputNorm(x), cache: kv)
        } else {
            fatalError("Drama 3 backbone cache does not match layer \(index)")
        }
        let h = x + mixed
        return h + mlp(postAttentionNorm(h))
    }
}

final class Drama3Backbone: Module {
    let config: Drama3Config.Backbone

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Drama3BackboneLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Drama3Config.Backbone) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.layerCount).map { Drama3BackboneLayer(config, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// Final-norm hidden states [B, L, hidden] for input embeddings; advances the caches.
    func callAsFunction(embeddings: MLXArray, cache: Drama3BackboneCache) -> MLXArray {
        var h = embeddings
        for (index, layer) in layers.enumerated() {
            h = layer(h, cache: cache, index: index)
        }
        cache.offset += embeddings.dim(1)
        return norm(h)
    }
}
