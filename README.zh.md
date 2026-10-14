# Yuwp

[English](README.md)

Yuwp 是一款 macOS 本地听写应用。按下快捷键后说话，即可在当前使用的应用中输入文字。

Yuwp 使用 Swift 和 MLX，在 Apple Silicon 上运行 Qwen3-ASR 或 Confucius4-R2T2。模型下载到本地后，音频不会离开 Mac。Yuwp 通过辅助功能 API 输入文字，在终端中使用键盘事件；如果这些方式失败，则回退到剪贴板。

## 系统要求

macOS 14+、Apple Silicon、Swift 6.4 / Xcode 27+。

## 安装

下载经过公证的版本：https://github.com/duh17/yuwp/releases/latest

从源码运行：

```bash
git clone https://github.com/duh17/yuwp.git
cd yuwp
xcodebuild -downloadComponent MetalToolchain   # once, on a fresh machine
scripts/run.sh
```

新机器只需执行一次上面的 Metal 工具链下载命令。`scripts/run.sh` 会为应用签名并启动 `/Applications/Yuwp.app`。

## 首次启动

1. 从菜单栏打开 Yuwp。
2. 点击 **Grant Accessibility Permission**，在「隐私与安全性」中允许 Yuwp 使用辅助功能。
3. 按一次快捷键，触发 macOS 的**麦克风**权限提示。
4. 在 **Settings… → Transcription** 中下载或选择模型。

## 使用

默认快捷键是 **Ctrl+`**。按一次开始听写，再按一次停止。

在 Settings 中可以更改快捷键、模型、服务器模式、录音、麦克风面板和提示音设置。

## 命令行工具与服务器

构建 ASR 和 TTS 工具：

```bash
swift build -c release --product yuwp-asr
swift build -c release --product yuwp-tts
```

DMG 和应用包均包含这两个工具：

```
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts
```

转写文件：

```bash
.build/out/Products/Release/yuwp-asr transcribe Tests/fixtures/jfk.wav
/Applications/Yuwp.app/Contents/MacOS/yuwp-asr transcribe Tests/fixtures/jfk.wav
```

启动 ASR HTTP 服务器（默认传输方式是 stdio，使用 HTTP 时需传入 `--transport http`）：

```bash
.build/out/Products/Release/yuwp-asr serve --model <asr-model-dir> --transport http --host 127.0.0.1 --port 7936
curl -sf http://127.0.0.1:7936/v1/info | jq .
```

`POST /v1/audio/transcriptions/stream` 创建会话，允许请求体为空。兼容的客户端可以发送 `{model, stream_config:{contextual_strings:[...]}}`。没有提示词时，必须省略 `stream_config`。

- 响应为 `{session_id, context_applied}`。只有使用了非空提示词，`context_applied` 才为 true。
- 限制：最多 100 个短语，每个短语最多 256 个 UTF-8 字节，总计最多 8192 字节。
- 不接受空字符串、仅含空白的字符串或含控制字符的字符串。
- 客户端不能设置 `system_prompt`；Yuwp 自行生成简短的词汇提示头。

批量转写会自动分块：短文件使用 VAD，超过 120 秒的文件按能量边界分块。`--batch-chunking automatic|vad|energy` 只影响批量转写，不影响实时流式转写。显式选择 `vad` 时，如果批量 VAD 无法加载，则回退到能量分块。旧的 `--disable-vad` 参数还会关闭流式 VAD。`GET /v1/info` 会报告请求的批量分块模式、实际采用的模式，以及批量 VAD 是否可用。

启动 TTS HTTP 服务器：

```bash
.build/out/Products/Release/yuwp-tts serve --transport http --model <qwen3-tts-model-dir> --host 127.0.0.1 --port 7937
curl -sf http://127.0.0.1:7937/v1/info | jq .
```

```bash
.build/out/Products/Release/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
/Applications/Yuwp.app/Contents/MacOS/yuwp-tts --model <qwen3-tts-model-dir> --text "Hello from Yuwp" --out /tmp/hello.wav
```

`POST /v1/audio/speech` 返回 WAV。`POST /v1/audio/speech/stream` 返回 NDJSON 事件（`metadata`、`audio`、`done`、`error`），其中音频块采用 base64 编码的 `pcm_s16le` 格式。

Qwen3-TTS 仍是日常低延迟默认。`yuwp-tts` 也可以加载原生 Swift/MLX 的 [AuK](https://github.com/Tencent-Hunyuan/AuK)：**AuK-Flash**（固定 4 步，关闭 CFG/sway）或 **AuK Base**（默认 `nfe=32`、`cfg=2.0`、`sway=-1.0`）。一个进程只加载一个模型目录；`GET /v1/info` 用 `backend`（`auk-flash` / `auk-base`）、`variant` 以及当前采样默认值报告。运行时不依赖 Python。

先把官方 PyTorch 权重转换一次。`--thinker-src` 可以是 Qwen2.5-Omni-3B，也可以是已转换的 mlx 目录（Base 可复用 Flash 的 thinker）：

```bash
.build/out/Products/Release/yuwp-tts convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK-Flash" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/Qwen2.5-Omni-3B" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx"

.build/out/Products/Release/yuwp-tts convert-auk \
  --src "$HOME/Library/Application Support/Yuwp/models/AuK" \
  --thinker-src "$HOME/Library/Application Support/Yuwp/models/auk-flash-mlx" \
  --out "$HOME/Library/Application Support/Yuwp/models/auk-base-mlx"
```

Instruct TTS 需要 `--gen-seconds`。克隆或编辑已有音频时传 `--ref-audio`。Base 接受 `--nfe --cfg --sway`（Flash 忽略）。

`POST /v1/audio/speech/stream` 是诚实流式：多 chunk 的 TTS 会在后续 chunk 开始生成之前写出第一段可播放 PCM。短句仍可能是一次完整 latent/VAE 解码，不能当成逐步流式。NDJSON：先 `metadata`（`pcm_s16le`、`encoding=base64`、`backend`、`variant`），再按完成的 TTS chunk 发 `audio`（`chunk` / `samples` / `seconds` / `elapsed_seconds` / `audio`），最后 `done`（`first_audio_seconds`、`audio_duration_seconds`、`wall_seconds`、`chunks`）。错误为 `event=error`。

长文本自动切分**只用于 TTS**：保留句子边界，每段套同一套音色/风格前缀，并在 chunk 之间插入 200 ms 静音（`--chunk-pause-ms`）。触发条件：`task`/`mode` 为 `tts`；或 `input` 是待说文本、`instruction` 是音色/风格；或 instruction 匹配官方 zero-shot/instruct 模板因而只切目标文本。已有音频的编辑 / 增强 / 分离保持「一条 instruction + 源音频」，对这些任务请求 auto-chunk 会结构化 400（`code=auto_chunk_unsupported`）。

Cookbook 式编辑（内容替换、音高/音量/情感、增强、分离）都走同一条路径。**已验证**与**仅可表达**：有权重时，内容替换与至少一种非内容变换会用 ASR / 音分 / dB 检查；歌词编辑可表达但不声称已验证。

### Fish Audio Drama 3（多角色、可控情绪的 TTS，仅命令行）

`yuwp-tts` 内置 [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) 的原生 Swift/MLX 实现：35B-A3B Qwen3.5-MoE 主干、13 层 codebook tower、44.1 kHz Fish codec。官方权重转换一次即可。转换器会反量化 block-FP8 专家，不用 Python 也能读取 `codec.pth`，并可选 8/4 bit 量化：

```bash
hf download fishaudio/drama-3
.build/out/Products/Release/yuwp-tts convert-drama3 \
  --src ~/.cache/huggingface/hub/models--fishaudio--drama-3/snapshots/<rev> \
  --out "$HOME/Library/Application Support/Yuwp/models/drama3-mlx-8bit" --bits 8

.build/out/Products/Release/yuwp-tts \
  --model "$HOME/Library/Application Support/Yuwp/models/drama3-mlx-8bit" \
  --text "[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair." \
  --ref-audio ref.wav --ref-text "ref.wav 的文字稿" \
  --out /tmp/drama3.wav
```

`--ref-audio` / `--ref-text` 可选（声音克隆）。重复多组即可给场景分配角色：第 *i* 组对应 `<|speaker:i|>`，旁白加四个角色就是五组；不给参考音频时由模型自己挑声音。`[laughs]` 等行内标签和 `<|speaker:N|>`（0–15）直接写在 `--text` 里。采样默认值与上游一致：`--temperature 1.0 --top-p 0.9 --top-k 30`；`--max-tokens` 限制帧数（每秒音频 21.5 帧）。`serve` 暂不支持 Drama 3。M3 Ultra 上 bf16（68 GB）约 21–29 帧/秒，8-bit（38 GB）约 35–37 帧/秒，4-bit（22 GB）约 43–52 帧/秒。权重使用 Fish Audio Research License，个人用途以外请先看许可证。

### YuE2 音乐生成（仅命令行）

`yuwp-music` 是 [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) 的原生 Swift/MLX 实现：输入风格标签和歌词，输出 48 kHz 立体声歌曲。YuE2 先写 ABC 谱（旋律，`full` 模式还带和弦），再据谱生成 25 Hz 语义码，用中点法 flow matching 求声学 latent，最后由 Oobleck VAE 解码。模型目录使用转换好的 [`vanch007/mlx-Yue2-3B`](https://huggingface.co/vanch007/mlx-Yue2-3B) 布局（`ar-bf16`、`ar-8bit`、`nar-bf16`、`qwen.tiktoken`、`vae/`），推理不用 Python。`--precision 8bit|4bit|bf16` 选择生成 token 的权重（默认 8-bit）；没有 `ar-4bit` 文件时，4-bit 会在加载时由 bf16 权重量化而来。flow matching 始终用 bf16。

```bash
hf download vanch007/mlx-Yue2-3B --local-dir ~/models/mlx-Yue2-3B
swift build -c release --product yuwp-music

.build/out/Products/Release/yuwp-music generate --model ~/models/mlx-Yue2-3B \
  --style "Mandarin, 1990s Hong Kong TVB wuxia theme, male tenor lead vocal, erhu, guzheng, strings" \
  --lyrics-file lyrics.txt --out /tmp/song
```

歌词每段单独一行写段落标签（`[Verse]`、`[Chorus]` 等）。`--mode full|melody|off` 选择谱的规划方式；`--abc-file` 直接渲染你自己的谱。`--steps` 设定 flow matching 步数（默认 32；8 步时这一阶段约快 4 倍）。输出目录包含 `audio.wav`、`score.abc`、`prefix.npy`、`semantic.npy`、`noise.npy`、`latent.npy` 和 `result.json`。`yuwp-music render --from <目录>` 用其他步数重解已存歌曲，`decode` 只跑 VAE。歌曲长度由模型决定（最长 360 秒）。同一 seed 在 `yuwp-music` 内可复现，但不会复现 mlx-Yue 的音频：两边的 MLX 版本对同一 key 抽样结果不同。YuE2 权重为 CC BY-NC 4.0，仅限个人与研究用途。

Mac Studio M3 Ultra，一首 4.4 分钟纯音乐：三个引擎使用同样的风格、谱、seed 和 32 步，token 权重均为 8-bit。各引擎各自采样，歌曲长度略有不同（括号内为帧数）。

| 阶段 | mlx-Yue（Python MLX） | mlx-serve 26.10.2（Zig） | `yuwp-music` |
| --- | ---: | ---: | ---: |
| 语义 token | 98.8 tok/s | 136.5 tok/s | **173.7 tok/s** |
| Flow matching，32 步 | 139.6 秒（6683） | 123.4 秒（6607） | **120.5 秒**（6497） |
| VAE 解码 | 5.4 秒 | 7.8 秒 | **2.0 秒** |
| 端到端 | 212.8 秒 | 181.6 秒 | **160.9 秒** |

按长度折算后，mlx-serve 与 `yuwp-music` 的 flow matching 基本持平（都用 bf16 MLX 对整段做注意力；mlx-Yue 约慢 10%）。差距主要在生成 token：`yuwp-music` 在 GPU 上采样，并在读回当前 token 之前就排好下一步。mlx-serve 的端到端不含加载模型（模型已常驻），另外两个包含。方法与原始数据见 [`benchmarks/`](benchmarks/README.md#yue2-music) 和 [`yue2-tvb-m3u.json`](benchmarks/fixtures/yue2-tvb-m3u.json)。4-bit token 权重可达 261 tok/s。

## 开发

```bash
swift build
swift test
scripts/build.sh
scripts/run.sh
```

Swift 6.4 的 SwiftPM 使用 Swift Build，二进制文件输出到 `.build/out/Products/{Debug,Release}`。`swift build` 需要 Metal 工具链（见「安装」）来编译 mlx-swift 着色器。

## 基准测试

两台机器均为 macOS 27.0（26A428）。复现命令：`uv run benchmarks/cli.py asr headline`。数字：[`benchmarks/fixtures/public-jfk.json`](benchmarks/fixtures/public-jfk.json)。

听写使用 100 ms stdio 分片，音频为 [`jfk.wav`](Tests/fixtures/jfk.wav)（11 s）。首段文字：能量起点（开头 0.30 s 静音）到第一条非空 partial。停止 → 最终：发送 stop 到最终转写。转写是同一条音频上 `yuwp-asr transcribe` 的墙钟时间。

云端流式分数（例如 Meta Muse Voice Transcribe，[Artificial Analysis](https://artificialanalysis.ai) 在 2026-09-01 给出约 0.16 s 出最终结果）从**语音结束**开始计时。那一列对应停止 → 最终，不是说话过程中的首字延迟。80 ms 是音频分片间隔，不是首字延迟。

M3U 是 Mac Studio（M3 Ultra，512 GB）。M4P 是 Mac mini（M4 Pro，64 GB）。

| 模型 | 首段 M3U | 首段 M4P | 停止 M3U | 停止 M4P | 转写 M3U | 转写 M4P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B 4-bit | 620 ms | 1210 ms | 94 ms | 176 ms | 1.18 s | 0.71 s |
| 0.6B bf16 | 618 ms | 1247 ms | 132 ms | 247 ms | 1.25 s | 0.84 s |
| 1.7B 4-bit | 625 ms | 1274 ms | 150 ms | 323 ms | 1.23 s | 0.90 s |
| 1.7B bf16 | 644 ms | 1375 ms | 181 ms | 383 ms | 1.55 s | 1.33 s |

M3U 上 1.7B bf16 的混合语音 Silero 起点 canary，首段文字为 505 ms。

**语音合成**，Qwen3-TTS 1.7B CustomVoice（M3U）：生成 4.16 s 音频耗时 2.05 s。

### R2T2

可选模型。当前 8-bit 权重是 `mlx-community/Confucius4-R2T2-8bit`：语言模型 8-bit affine（group 64），音频编码器仍是 bf16，不是 FP8。Settings 里还有未量化的 bf16（4.08 GB）。

计时口径与上面的 Qwen 表相同。机器是 Mac Studio M3 Ultra，macOS 27.2（26B5091g）。尚未测 Mini。复现：`uv run benchmarks/cli.py asr headline --skip-batch --skip-tts --asr-model LABEL=PATH`。数字：[`benchmarks/fixtures/public-jfk-r2t2.json`](benchmarks/fixtures/public-jfk-r2t2.json)。

| 模型 | 首段 M3U | 停止 M3U | 转写 M3U |
| --- | ---: | ---: | ---: |
| R2T2 8-bit | 990 ms | 171 ms | 0.86 s |
| R2T2 bf16 | 939 ms | 206 ms | 1.03 s |

R2T2 每 160 ms 提交一次已确认文本，并扣住最后一个 token，所以这条 headline 的首字会晚于 Qwen 的 rollback/preview。提交之后能跟上分片：`jfk.wav` 上 Silero 语音起点到首字是 800 ms；有语音的 160 ms 分片中位数 47 ms，p99 60 ms。这段 11 秒音频的停止→最终是 171 ms，其中包含短会话的 stop-batch。`jfk.wav` 和 `asr_zh.wav` 上流式与 batch 的 WER 为 0。

官方 [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2) 公布了数据集 WER/CER，以及 CUDA vLLM / llama.cpp 上 200–600 ms 的平均流式延迟（retrospective chunk-wise mean fuzzy latency）。没有 Apple Silicon 数字。那条 Python 路径需要 NVIDIA CUDA，这台 Mac 跑不了。上表是 Yuwp 的 MLX Swift 移植，不是同一硬件上与 vLLM 的对比。

更多场景见 [`benchmarks/`](benchmarks/README.md)。

## 隐私

音频在 Mac 本地处理。默认不录音，只有开启 **Save Recordings** 才会保存。诊断日志默认关闭，可在 Settings 中打开。

## 致谢

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)
- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [qwen-asr](https://github.com/antirez/qwen-asr)（流式处理的参考思路）
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3) 与 [SGLang-Omni](https://github.com/sgl-project/sglang-omni)（Drama 3 推理参考）
- [YuE2](https://github.com/multimodal-art-projection/YuE)、[mlx-Yue](https://github.com/vanch007/mlx-Yue) 与 [mlx-serve](https://github.com/ddalcu/mlx-serve)（YuE2 推理参考）

## 许可证

- [MIT](LICENSE)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- 应用包和发布的 DMG 均包含这些声明，以及 `OpenSource/` 目录下随包分发的上游许可证。
