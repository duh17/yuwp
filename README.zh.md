# Yuwp

[English](README.md)

Yuwp 是一款 macOS 本地听写应用。按下快捷键后说话，即可在当前使用的应用中输入文字。

Yuwp 使用 Swift 和 MLX，在 Apple Silicon 上运行 Qwen3-ASR。模型下载到本地后，音频不会离开 Mac。Yuwp 通过辅助功能 API 输入文字，在终端中使用键盘事件；如果这些方式失败，则回退到剪贴板。

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

更多场景见 [`benchmarks/`](benchmarks/README.md)。

## 隐私

音频在 Mac 本地处理。默认不录音，只有开启 **Save Recordings** 才会保存。诊断日志默认关闭，可在 Settings 中打开。

## 致谢

- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [MLX](https://github.com/ml-explore/mlx) 和 [mlx-swift](https://github.com/ml-explore/mlx-swift)
- [qwen-asr](https://github.com/antirez/qwen-asr)（流式处理的参考思路）
- [Silero VAD](https://github.com/snakers4/silero-vad)

## 许可证

- [MIT](LICENSE)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- 应用包和发布的 DMG 均包含这些声明，以及 `OpenSource/` 目录下随包分发的上游许可证。
