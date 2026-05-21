# 技术文档 — 中文实时语音聊天应用

## 1. 项目概述

一个支持文字输入和语音输入的中文 AI 实时聊天应用，包含两套界面：

- **命令行版** ([chatbot.py](chatbot.py))：终端多轮对话，使用 `sounddevice` 在本地扬声器播放 AI 回复。
- **网页版** ([web_server.py](web_server.py) + [static/](static/))：浏览器单页应用，支持麦克风/键盘输入，AI 文字逐字流式显示、语音边生成边在浏览器播放，可在 AI 说话时按麦克风按钮打断（barge-in）。

两套界面共享同一套 LLM 调用和 TTS 调用逻辑，网页版额外集成本地 ASR。

### 核心功能

| 功能 | 说明 |
|---|---|
| 多轮中文对话 | LLM 维护对话历史，跨轮上下文连贯 |
| 流式文字输出 | LLM token 一到即推送到前端，气泡内逐字显示 |
| 流式语音合成 | 一拿到完整句子立刻触发 TTS，与下一句并行 |
| 浏览器麦克风输入 | 点击按钮录音、再点结束，整段送服务端识别 |
| 离线中文 ASR | 服务端 FunASR `paraformer-zh` + `ct-punc`，无外部依赖 |
| Barge-in | 点击麦克风按钮立即停止当前 AI 播放，进入新一轮录音 |
| 思维链可视化 | LLM 的 `thinking_delta` 单独以灰色斜体展示，不参与 TTS |

---

## 2. 技术栈

### 后端 (Python 3.12)

| 库 | 作用 |
|---|---|
| `fastapi` + `uvicorn[standard]` | Web 框架 + ASGI 服务器 |
| `anthropic` (AsyncAnthropic) | LLM 调用（指向 MiniMax 的 Anthropic 兼容端点） |
| `httpx` | TTS HTTP SSE 流式调用 |
| `funasr` 1.3.1 | 中文 ASR（paraformer-zh）+ 标点（ct-punc） |
| `av` (PyAV) | 浏览器送来的 WebM/Opus 解码到 PCM |
| `torch` + `torchaudio` (CPU) | FunASR 底层 |
| `numpy` | PCM/特征数据处理 |
| `sounddevice` | CLI 版本地播放（网页版不用） |
| `requests` | CLI 版 TTS 调用（网页版不用） |
| `python-dotenv` | 从 `.env` 加载 API key |

### 前端 (纯静态 HTML + ES Module JS，无框架)

| API | 作用 |
|---|---|
| WebSocket | 与服务端双向通信，文本 JSON + 自描述二进制帧 |
| `MediaRecorder` | 麦克风采集 → WebM/Opus 整段 |
| Web Audio API (`AudioContext` + `AudioBufferSourceNode`) | PCM 块排队、无缝衔接播放 |

### 外部服务

- **LLM**：MiniMax `MiniMax-M2.7`，端点 `https://api.minimaxi.com/anthropic`（Anthropic SDK 兼容），流式。
- **TTS**：MiniMax `https://api.minimaxi.com/v1/t2a_v2`，模型 `speech-2.8-hd`，音色 `male-qn-qingse`，SSE 流式，PCM 32 kHz mono int16 hex 编码。

### 本地模型

- FunASR 在首次运行时从 ModelScope 下载到 `./models`（约 1.2 GB）：
  - `paraformer-zh` ~840 MB（中文 ASR）
  - `ct-punc` ~280 MB（标点恢复）
  - `fsmn-vad` ~5 MB（FunASR 隐式依赖）

---

## 3. 文件结构

```
chatbot/
├── chatbot.py          # CLI 版主程序（保持原样，参考实现）
├── config.py           # 共享常量与 .env 加载
├── web_server.py       # FastAPI 入口，WebSocket /ws 端点，per-conn 状态机
├── web_llm.py          # 异步 LLM 流式封装
├── web_tts.py          # 异步 TTS SSE 流式封装 → 产 PCM bytes 异步迭代器
├── web_asr.py          # FunASR + ct-punc 加载、PyAV 解码、文本后处理
├── static/
│   ├── index.html      # 单页 UI
│   └── app.js          # WebSocket 客户端 + MediaRecorder + Web Audio
├── test_llm.py         # CLI 时期 LLM 冒烟测试（手动跑）
├── test_tts.py         # CLI 时期 TTS 冒烟测试（手动跑）
├── requirements.txt    # 依赖清单（含 torch CPU 安装说明）
├── .env                # MINIMAX_API_KEY、ANTHROPIC_BASE_URL（不入仓）
├── .venv/              # conda Python 3.12 环境（不入仓）
├── models/             # FunASR 模型缓存（不入仓）
├── CLAUDE.md           # AI 助手指南（landmine 提醒）
└── DESIGN.md           # 本文档
```

### 模块职责

- **[config.py](config.py)**：唯一加载 `.env` 的地方，集中导出 `API_KEY`、`LLM_BASE_URL`、`TTS_URL`、`LLM_MODEL`、`TTS_MODEL`、`TTS_VOICE`、`SAMPLE_RATE_TTS=32000`、`SAMPLE_RATE_ASR=16000`、`SENTENCE_ENDS="。！？!?；;\n"`、`SYSTEM_PROMPT`、`MODELS_DIR`。`load_dotenv(override=True)` 让 `.env` 覆盖 shell 环境（关键，否则系统级 `ANTHROPIC_BASE_URL=https://api.anthropic.com` 会污染）。

- **[web_llm.py](web_llm.py)**：`stream_reply(messages, on_text_delta, on_thinking_delta)`，用 `anthropic.AsyncAnthropic.messages.stream(...)` 上下文管理器（确保 cancel 时 HTTP 连接释放）。对外只暴露增量回调。

- **[web_tts.py](web_tts.py)**：`async def synthesize(text) -> AsyncIterator[bytes]`，用 `httpx.AsyncClient.stream("POST", ...)` 拉 SSE，每行解析 JSON，提取 hex 编码 PCM 解码后 yield。**显式跳过 `status==2` 的事件**（MiniMax 在流末尾会再发一份完整音频拼接，不跳过会重复播放）。

- **[web_asr.py](web_asr.py)**：
  - `load_model()`：单例加载 `AutoModel(model="paraformer-zh")` 和 `AutoModel(model="ct-punc")`，跑 warmup。
  - `transcribe(audio_bytes)`：通过 magic bytes (`0x1A45DFA3` for WebM EBML / `RIFF` / `OggS`) 判定输入是裸 PCM 还是容器格式；容器格式经 PyAV `av.open()` + `AudioResampler` 转 16 kHz mono PCM。ASR 后调 `_punc_model.generate()` 补标点。
  - `DEBUG_ASR=1` 环境变量启用 → 每次输入 dump 到 `./debug_asr.wav`。

- **[web_server.py](web_server.py)**：见下节"运行时架构"。

- **[static/app.js](static/app.js)**：WebSocket 连接（带指数退避自动重连）、`MediaRecorder` 麦克风采集、二进制帧解析与 PCM 播放调度、barge-in 处理。

---

## 4. 运行时架构

### 4.1 系统架构图

```
┌─────────────────────────────────────────┐         ┌──────────────────────────────────────────────┐
│ Browser (Chrome)                        │         │ Server (FastAPI, asyncio)                    │
│                                         │         │                                              │
│  ┌─────────────┐    ┌──────────────┐    │         │   ┌─────────────────────────────────────┐    │
│  │ index.html  │    │ app.js (ES)  │    │         │   │ web_server.py                       │    │
│  │  - 对话区   │    │              │    │         │   │  GET /        → index.html          │    │
│  │  - 输入框   │    │  WebSocket   │◄───┼────WS───┼──►│  GET /static  → static files        │    │
│  │  - 麦克风   │    │  MediaRec.   │    │         │   │  WS  /ws      → ws_endpoint         │    │
│  └─────────────┘    │  WebAudio    │    │         │   └────┬─────────────┬──────────────────┘    │
│                     └──────────────┘    │         │        │             │                       │
└─────────────────────────────────────────┘         │        ▼             ▼                       │
                                                    │   ConnCtx (per-WS)   lifespan startup       │
                                                    │    - messages[]       └─► web_asr.load     │
                                                    │    - turn_id                                │
                                                    │    - current_task     ┌──────────────────┐  │
                                                    │    - outbound queue   │ web_asr.py       │  │
                                                    │    - sender_loop      │  FunASR + ct-punc│  │
                                                    │                       │  PyAV decode     │  │
                                                    │   _handle_turn        └──────────────────┘  │
                                                    │    │   ┌──────────────┐ ┌──────────────┐    │
                                                    │    │   │ web_llm.py   │ │ web_tts.py   │    │
                                                    │    └──►│ AsyncAnthrop.│►│ httpx SSE    │    │
                                                    │        │  stream      │ │  → PCM bytes │    │
                                                    │        └──────────────┘ └──────────────┘    │
                                                    │           │                  │              │
                                                    │           ▼                  ▼              │
                                                    │      MiniMax LLM        MiniMax TTS         │
                                                    └──────────────────────────────────────────────┘
```

### 4.2 一轮对话的执行流程

```
用户点麦克风
   │
   ▼
[browser] MediaRecorder.start()  ─JSON─►  {type:"audio_start"}
                                          │
                                          ▼
                                  [server] ctx.cancel_current()  (取消上一轮)
                                          ctx.audio_buf.clear()
   │
用户再点麦克风
   ▼
[browser] MediaRecorder.stop()
          收集 dataavailable 的 Blob 列表
          concat → arrayBuffer
   │
   ├─binary─►  [server] ctx.audio_buf.extend(bytes)
   │
   └─JSON─►   {type:"audio_end"}
                           │
                           ▼
              [server] _start_new_turn(audio_runner)
                       ↳ turn_id++; create_task(_handle_audio)
                                │
                                ▼
                      _handle_audio:
                       1. PyAV decode (WebM/Opus → PCM 16k mono float32)
                       2. FunASR paraformer-zh → "你 好 今 天 ..."
                       3. ct-punc.generate → "你好，今天..."
                       4. send {type:"asr_result", text, turn_id}
                       5. _handle_turn(asr_text)
                                │
                                ▼
                      _handle_turn:
                        TaskGroup:
                          ├─ llm_producer:
                          │    async with stream(...) as s:
                          │      for event in s:
                          │        if text_delta:
                          │           send {type:"llm_delta"}
                          │           buffer += text
                          │           按 SENTENCE_ENDS 切句 → sentence_queue.put((seq, sentence))
                          │        if thinking_delta:
                          │           send {type:"thinking_delta"}
                          │    finally: 推 sentinel；send {type:"llm_done"}
                          │
                          └─ tts_consumer:
                               for (seq, sentence) in sentence_queue:
                                 async for pcm in synthesize(sentence):
                                   send make_audio_frame(turn_id, seq, 0, pcm)
                                 send make_audio_frame(turn_id, seq, FLAG_FINAL, b"")

       浏览器:
         JSON  → 更新对话气泡 / 状态指示
         二进制 → 校验 magic + turn_id  → Int16Array → AudioBuffer
                  → AudioBufferSourceNode.start(nextStartTime)
                  → nextStartTime += duration  (gapless 调度)
```

### 4.3 Barge-in 流程

```
[AI 正在说话]
        │
用户点麦克风
        │
[browser] stopAllPlayback()            ← 立即停所有 AudioBufferSourceNode、清空 nextStartTime
          MediaRecorder.start()
          ─JSON─► {type:"audio_start"}
                              │
[server] ctx.cancel_current()  ← 取消 current_task：
                                   - TaskGroup 抛 BaseExceptionGroup[CancelledError]
                                   - llm_producer 的 async-with stream(...) 退出，关闭 HTTP
                                   - tts_consumer 的 synthesize() async-for 中断
                                   - except* CancelledError: pass
                                   - finally: 若 assistant_text 为空，messages.pop() 回滚 user 消息
         turn_id 在 _start_new_turn 时 ++
         
旧 turn_id 的 TTS 帧若已在 ws.send 排队 → 浏览器收到时按 turn_id 过滤丢弃
```

---

## 5. WebSocket 协议

### 5.1 文本消息 (UTF-8 JSON)

**Client → Server**

| `type` | 字段 | 说明 |
|---|---|---|
| `text_input` | `text: string` | 文字提交 |
| `audio_start` | — | 用户开始录音，服务端取消现有轮次并清空 audio_buf |
| `audio_end` | — | 用户结束录音，触发 ASR |
| `cancel` | — | 显式取消（前端目前没用，保留扩展点） |

**Server → Client**

| `type` | 字段 | 说明 |
|---|---|---|
| `asr_result` | `text, turn_id` | ASR 结果（已带标点） |
| `thinking_delta` | `text, turn_id` | LLM 思维链增量（不进 TTS，UI 灰色显示） |
| `llm_delta` | `text, turn_id` | LLM 回复增量（进 TTS） |
| `llm_done` | `turn_id` | LLM 流结束 |
| `tts_sentence_done` | `seq, turn_id` | 一句 TTS 全部发完（诊断用） |
| `error` | `where, message` | `where` ∈ `asr`/`llm`/`tts` |

### 5.2 二进制帧（仅 Server → Client）

```
┌──────────────────────────────────────────────────────┬─────────────┐
│ Header (16 bytes, little-endian)                     │ Payload     │
├──┬──┬──┬─────────┬─────────┬────────┬────────────────┤             │
│M0│M1│Ver│ turn_id │   seq   │ flags  │   reserved    │ PCM int16 LE│
│A1│B2│01 │ uint32  │ uint32  │ uint8  │   4 bytes     │ mono 32 kHz │
├──┴──┴──┴─────────┴─────────┴────────┴────────────────┴─────────────┘
   0  1  2     3       7         11           12        16 ...
```

- `flags & 1` = "本帧是 `seq` 这句话的最后一帧"
- 前端按 magic + version 校验帧合法性；按 `turn_id == currentTurnId` 过滤过期帧
- 长度 < 16 字节或 payload 为空 + final flag → "句末标记"，不送播放

构造点：[web_server.py](web_server.py) 的 `make_audio_frame()`。解析点：[static/app.js](static/app.js) 的 `handleBinaryFrame()`。

---

## 6. 关键技术决定

### 6.1 为何 MediaRecorder 而非 AudioWorklet

最初前端用 `AudioWorklet` + 手写降采样把麦克风 Float32@48kHz 转 Int16@16kHz，逐 quantum 发送 PCM 块。**实测在第二次及之后的录音上 Chrome 调度不稳，产生周期性振幅调制（颤音）**，FunASR 输出垃圾。即便加上 `muted GainNode → destination` 保持图被拉取，问题仍存在。

切到 `MediaRecorder` + WebM/Opus 整段发送：
- 走 Chrome 自带的 native media pipeline（与视频会议同源），跳过 AudioWorklet 调度
- 容器自带正确时间戳，无丢/重复帧
- 服务端用 PyAV（自带 libffmpeg，无需系统 ffmpeg）解码到 PCM 16kHz mono float32 喂 FunASR

代价：录音中无法流式发；但当前交互是"点击-停止"，本来就只在结束时识别，无影响。

### 6.2 句级 LLM→TTS 流水线

LLM 文本一边到一边累积，一遇到句末标点 (`。！？!?；;\n`) 立刻把整句推到 `sentence_queue`。TTS consumer 串行（concurrency=1）拉句子调 MiniMax TTS，保证音频按句子顺序输出，前端用 `nextStartTime += duration` gapless 调度。

并发=1 而非更高，是为了：
- 避免前端排序复杂度（一句一句来）
- 不至于把短句的首字延迟拉长（TTS API 对长句首块延迟更敏感）

### 6.3 异步取消的正确姿势

- **LLM**：必须 `async with client.messages.stream(...) as s:`，不能裸 `messages.create(stream=True)`。后者在 `task.cancel()` 时不会关闭底层 HTTP 连接，长跑会泄漏 socket。
- **TaskGroup**：把 `llm_producer` 和 `tts_consumer` 放进 `asyncio.TaskGroup`，外层 task cancel 时 TaskGroup 的 `__aexit__` 会级联取消所有子任务并等待清理完成。
- **`except* CancelledError`**：3.11+ 用 except-star 捕获 TaskGroup 抛出的 `BaseExceptionGroup`，避免 cancel 误染色为异常。
- **历史回滚**：放在 `finally` 而非 `except` 分支，确保 cancel 路径也走到。

### 6.4 ct-punc 必须独立加载

`AutoModel(model="paraformer-zh", punc_model="ct-punc")` 在 FunASR 1.3.1 上**只加载** ct-punc 但**不在 `generate()` 时自动调用**——返回的 `text` 是 `"你 好 今 天"` 这种字间空格、无标点的形式。

解决：单独 `_punc_model = AutoModel(model="ct-punc", ...)`，ASR 后手动 `_punc_model.generate(input=raw_text)`。失败时兜底为 `text.replace(" ", "")` 至少保证可读。

### 6.5 .env 必须 override 系统环境

Windows 上常有 shell 级 `ANTHROPIC_BASE_URL=https://api.anthropic.com` 残留。`load_dotenv()` 默认不覆盖已存在环境变量 → MiniMax 客户端跑去打 Anthropic 官网，得 401。改为 `load_dotenv(override=True)`。

### 6.6 二进制帧自描述

每个音频帧前 16 字节自带 magic + turn_id + seq + flags，不依赖前面的 JSON header 描述。原因：
- WebSocket 虽保序，但生产者侧 cancel 可能让 JSON header 和 binary payload 失配
- 前端在 barge-in 后可能还会收到旧 turn 的 in-flight 帧 — 用 turn_id 自描述直接丢弃

---

## 7. 延迟预算

mic-stop → 首段音频实测在 1.5–3.5 秒之间，主要分布：

| 阶段 | 估计 |
|---|---|
| MediaRecorder stop + Blob.arrayBuffer + WebSocket 上传 | 50–200 ms |
| PyAV 解码 + 重采样 | 30–100 ms |
| FunASR paraformer 推理 | 300–800 ms（CPU，依输入长度） |
| ct-punc 标点 | 30–80 ms |
| LLM TTFT（MiniMax） | 500–1500 ms（最大变量） |
| 首句切分等待 | 100–300 ms |
| TTS TTFB（MiniMax） | 300–600 ms |
| 浏览器播放预缓冲（`PLAYBACK_LEAD_SEC = 0.2`） | 200 ms |

主要瓶颈是外部 LLM TTFT 和本地 FunASR CPU 推理。

---

## 8. 部署与运行

### 8.1 首次环境搭建

```powershell
# Python 3.12 conda env（不要用 3.13）
conda create -p ./.venv python=3.12 -y

# PyTorch CPU（从专用索引，不要混进 requirements.txt）
.venv/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 其余依赖
.venv/python.exe -m pip install -r requirements.txt

# 准备 .env（仓库不含）
@"
MINIMAX_API_KEY=<你的 MiniMax key>
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
"@ | Out-File -Encoding utf8 .env
```

### 8.2 日常启动

```powershell
# Web 版（首次启动会下载 ~1.2GB FunASR 模型，需 5–15 分钟）
.venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000

# CLI 版
.venv/python.exe chatbot.py
```

浏览器打开 `http://127.0.0.1:8000`（**必须用 `127.0.0.1` 或 `localhost`** — 浏览器要求 secure context 才会开放 `getUserMedia`，LAN IP 会被拒）。

### 8.3 调试

```powershell
# Dump 每次 ASR 输入到 debug_asr.wav，用任意播放器听
$env:DEBUG_ASR=1; .venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000
```

前端调试看 Chrome DevTools Console，关键日志：`[mic] MediaRecorder mimeType = audio/webm;codecs=opus` 和 `[mic] sending N bytes, ...`。

---

## 9. 可扩展方向

短期优化：
- 服务端 send 加 outbound 队列 maxsize 限流（已做 `maxsize=200`，需观察实际触发情况）
- 浏览器播放 lead 时间自适应（jitter 大时拉长，jitter 小时缩短）
- 前端把 thinking 区域改成可折叠

中期：
- 支持 push-to-talk 与 VAD 持续监听两种模式切换
- 加上对话历史持久化（localStorage 或 sqlite）
- 多用户/多会话隔离（当前实现假设单连接 = 单会话）

长期：
- 替换 LLM provider（接 OpenAI / Anthropic / 通义千问，复用句切分流水线）
- ASR/TTS 多语言扩展（SenseVoice + 多语 TTS）
- 客户端打包为 Electron / Tauri 桌面应用
