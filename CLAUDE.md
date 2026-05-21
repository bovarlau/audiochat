# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

中文语音聊天应用，两套界面共享 LLM + TTS 后端：

- **CLI** ([chatbot.py](chatbot.py)) — 终端多轮对话，`sounddevice` 本地播放。**不要改动**，是参考实现。
- **Web** ([web_server.py](web_server.py) + [static/](static/)) — 浏览器 UI，麦克风/文字输入，PCM 流回浏览器播放，支持 barge-in。

LLM 和 TTS 都走 MiniMax 的 Anthropic 兼容端点（`api.minimaxi.com`），ASR 用本地 FunASR `paraformer-zh` + `ct-punc`。

**完整架构、协议、设计回溯、延迟预算见 [DESIGN.md](DESIGN.md)**。本文件只列容易踩坑的事，避免改坏。

## 项目状态与有意不做的事

单人开发的语音聊天项目，**没有** 也**不要主动加**以下东西（用户没要的话）：

- 没有 git 仓库 / `.gitignore`
- 没有 pytest / 单元测试框架（`test_llm.py`、`test_tts.py` 只是手动冒烟脚本）
- 没有 CI / lint / formatter 配置
- 没有 Docker / 容器化
- 没有结构化 logging（`print(..., file=sys.stderr)` 足够）
- 前端没有打包步骤（纯 ES module，浏览器直接加载 [static/app.js](static/app.js)）
- `requirements.txt` 不固定版本号（除注释里说明 torch CPU 索引外）

加任何一项前都先问用户。

## 环境

- **Python 3.12**（conda env at `.venv`）。**不要升级到 3.13** — `funasr` 依赖 `editdistance` 在 3.13 上没有预编译 wheel，源码编译在当前 MSVC 失败。
- `.venv` 是 conda 环境，Python 在 `.venv/python.exe`（根目录），scripts 在 `.venv/Scripts/`。不要用 `.venv/bin/...` 这种 POSIX 路径。
- `.env` 提供 `MINIMAX_API_KEY` 和可选的 `ANTHROPIC_BASE_URL`。[config.py](config.py) 用 `load_dotenv(override=True)` —— shell 环境里若已有 `ANTHROPIC_BASE_URL=https://api.anthropic.com`（很常见）会覆盖 .env，必须 override。

## 常用命令

```powershell
# CLI 版
.venv/python.exe chatbot.py

# Web 服务（127.0.0.1 only，不要暴露 LAN — getUserMedia 需要 secure context）
.venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000

# 调试：dump 每次 ASR 输入到 ./debug_asr.wav
$env:DEBUG_ASR=1; .venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000

# 模块语法检查
.venv/python.exe -m py_compile config.py web_asr.py web_llm.py web_tts.py web_server.py
```

依赖重装（重建 conda env 后）：
```powershell
# 1. PyTorch CPU 必须先于 requirements.txt（用专门的索引）
.venv/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# 2. 其余依赖
.venv/python.exe -m pip install -r requirements.txt
```

## 架构要点

### 数据流（Web 端一轮对话）

```
浏览器                                  服务端
─────────                              ───────────────────────
[文字输入] ─JSON─►                      _handle_turn:
                                         LLM stream (web_llm.AsyncAnthropic)
                                          ├─ text_delta → ws json
                                          ├─ thinking_delta → ws json
                                          └─ 按 SENTENCE_ENDS 切句
                                                 ▼
                                         sentence_queue (asyncio)
                                                 ▼
                                         TTS stream (web_tts.httpx SSE)
                                          → PCM 块 → 自描述二进制帧 → ws

[🎙 录音] ─MediaRecorder→ WebM/Opus ─►   _handle_audio:
                                         PyAV decode → PCM 16k mono
                                          → web_asr.transcribe (FunASR + ct-punc)
                                          → 同 _handle_turn 流程
```

### 关键设计决定（踩过坑的）

1. **不要用 AudioWorklet 采集麦克风**。早期版本用 AudioWorkletProcessor 做 48k→16k 降采样，在第二次及之后的录音上 Chrome 调度不稳产生颤音，FunASR 输出垃圾。当前方案：浏览器 `MediaRecorder` → WebM/Opus 整段发送 → 服务端 PyAV 解码。如果有人想"优化为流式 mic"，看 git 历史，别再走 AudioWorklet 路。

2. **FunASR 1.3.1 的 `punc_model=` 参数加载但不自动调用**。`AutoModel.punc_model` 是裸 `CTTransformer` nn.Module，没有 `.generate()`。[web_asr.py](web_asr.py) 单独加载 `_punc_model = AutoModel(model="ct-punc")`，在 ASR 后手动调用。删除这层后处理 → 输出变成 `"你 好 今 天"` 这种字间带空格、无标点的形式。

3. **TTS SSE `status==2` 必须跳过**。MiniMax `/v1/t2a_v2` 流式末尾会发一个 `status==2` 的事件，里面 `audio` 字段是整段拼接好的 PCM。不跳过 → 每句话播两遍。见 [web_tts.py](web_tts.py)。

4. **LLM 取消必须用 `async with messages.stream()` 上下文管理器**，不是裸 `messages.create(stream=True)` —— 否则 cancel 时 HTTP 连接泄漏。`asyncio.TaskGroup` 用来把取消传播到所有 TTS 子任务。见 [web_server.py:167-181](web_server.py)。

5. **二进制帧前 16 字节是 header**（magic `0xA1 0xB2`、version、turn_id、seq、flags、reserved）。前端按 `turn_id` 过滤旧 turn 的延迟帧（barge-in 后必须）。`make_audio_frame()` 是唯一构造点，前端在 [static/app.js](static/app.js) 的 `handleBinaryFrame()` 解析。

6. **历史回滚**。如果一轮没产生任何 assistant text（被 cancel 或异常），从 `messages` 弹出本轮 user message — 否则历史里会悬空一条 user。CLI 和 Web 都有这逻辑。

### 文件依赖

```
config.py ──► web_llm.py  ──┐
         ├──► web_tts.py  ──┤
         └──► web_asr.py  ──┴──► web_server.py ──► static/
```

`config.py` 是唯一允许调 `load_dotenv()` 的地方。其他模块只从 config 导入常量。

## ASR 模型

首次运行时 FunASR 会从 ModelScope 下载到 `./models`（约 1.2GB：paraformer-zh ~840MB + ct-punc ~280MB + fsmn-vad ~5MB）。`./models` 已加入运行时缓存，不要 commit 到仓库。

`web_asr.load_model()` 在 FastAPI lifespan 里以 `asyncio.to_thread` 启动，跑 1s 静音 warmup 消除首句冷启动惩罚。模型单例全局复用，转录用 `asyncio.to_thread` 调用。

## 浏览器要求

- Chrome（其他浏览器没测）。访问 `http://127.0.0.1:8000` 或 `http://localhost:8000`。**不要用 LAN IP** — `getUserMedia` 在非 secure context 上会被拒。
- 改 [static/app.js](static/app.js) 或 [static/index.html](static/index.html) 后需要 Ctrl+Shift+R 强制刷新（普通刷新有时拿不到新版本）。

## 测试脚本

[test_llm.py](test_llm.py) 和 [test_tts.py](test_tts.py) 是 CLI 模式时期的快速冒烟脚本，直接 `python test_llm.py` / `python test_tts.py` 运行。它们没有 pytest harness，只是手动验证 API 接通。
