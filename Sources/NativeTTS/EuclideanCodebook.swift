import MLX
import MLXNN

final class EuclideanCodebook: Module {
    private let epsilon: Float = 1e-5
    private let dim: Int

    var initialized: MLXArray
    var embedding_sum: MLXArray
    var cluster_usage: MLXArray

    private var _embedding: MLXArray
    private var _c2: MLXArray

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.initialized = MLXArray.zeros([1], dtype: .float32)
        self.embedding_sum = MLXArray.zeros([codebookSize, dim], dtype: .float32)
        self.cluster_usage = MLXArray.zeros([codebookSize], dtype: .float32)

        let safeUsage = MLX.maximum(cluster_usage, MLXArray(epsilon)).reshaped([codebookSize, 1])
        self._embedding = embedding_sum / safeUsage
        self._c2 = _embedding.square().sum(axis: -1) / 2
    }

    private func refreshDerivedState() {
        let safeUsage = MLX.maximum(cluster_usage, MLXArray(epsilon)).reshaped([cluster_usage.shape[0], 1])
        _embedding = embedding_sum / safeUsage
        _c2 = _embedding.square().sum(axis: -1) / 2
    }

    override func update(parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [], modulePath: [String] = []) throws -> Self {
        try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        refreshDerivedState()
        return self
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        let targetShape = xs.shape + [dim]
        let taken = MLX.take(_embedding, xs.flattened(), axis: 0)
        return taken.reshaped(targetShape)
    }
}
