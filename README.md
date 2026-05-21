# 中文实时语音聊天应用

一个支持文字和语音输入的中文 AI 实时聊天应用,提供命令行和网页两套界面。网页版可以用麦克风说话或直接打字,AI 的回复会逐字显示并实时合成语音在浏览器播放,还能在 AI 说话时随时打断它(barge-in)。

对话(LLM)和语音合成(TTS)使用 MiniMax 云端服务,语音识别(ASR)使用本地运行的 FunASR 中文模型,识别过程不依赖外部网络。

## 功能特性

| 功能 | 说明 |
|---|---|
| 多轮中文对话 | 维护对话历史,跨轮上下文连贯 |
| 流式文字输出 | AI 回复逐字实时显示 |
| 流式语音合成 | 整句一生成即合成播放,边说边出 |
| 浏览器麦克风输入 | 点击录音,整段送服务端识别 |
| 本地中文语音识别 | FunASR 离线识别 + 自动标点,无需联网 |
| 打断(barge-in) | AI 说话时可随时打断并开始新一轮 |

## 两套界面

- **命令行版**(`chatbot.py`)：在终端里进行多轮对话,AI 回复通过本地扬声器播放。
- **网页版**(`web_server.py` + `static/`)：浏览器单页应用,支持麦克风和键盘输入。

## 环境要求

- **Python 3.12**(请勿使用 3.13,部分依赖在 3.13 上没有预编译包、安装会失败)
- Windows + PowerShell(下方命令以此为准)
- 一个 **MiniMax API Key**(用于 LLM 与 TTS)
- 网页版需使用 **Chrome** 浏览器

## 安装

建议用 conda 在项目内创建独立环境。在项目根目录依次执行:

```powershell
# 1. 创建 Python 3.12 环境
conda create -p ./.venv python=3.12 -y

# 2. 先安装 PyTorch CPU 版（必须用专用索引，且要先于第 3 步）
.venv/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装其余依赖
.venv/python.exe -m pip install -r requirements.txt
```

## 配置

在项目根目录创建 `.env` 文件,填入你的 MiniMax 凭据:

```
MINIMAX_API_KEY=你的_MiniMax_API_Key
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
```

> `.env`、`.venv/`、`models/` 均已在 `.gitignore` 中忽略,不会上传到仓库。

## 运行

### 网页版

```powershell
.venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000
```

启动后用浏览器打开 **http://127.0.0.1:8000** 即可对话。

> **首次启动**会自动从 ModelScope 下载约 1.2GB 的 FunASR 中文模型(约需 5–15 分钟),完成后才能正常识别语音。模型缓存在 `./models`,之后再启动无需重新下载。
>
> 必须用 `127.0.0.1` 或 `localhost` 访问 —— 浏览器只在安全上下文下开放麦克风权限,用局域网 IP 会被拒绝。

### 命令行版

```powershell
.venv/python.exe chatbot.py
```

## 调试(可选)

设置 `DEBUG_ASR=1` 后启动,会把每次语音识别的输入音频保存为 `./debug_asr.wav`,方便排查识别问题:

```powershell
$env:DEBUG_ASR=1; .venv/Scripts/uvicorn.exe web_server:app --host 127.0.0.1 --port 8000
```

## 项目结构

```
chatbot/
├── chatbot.py        # 命令行版主程序
├── web_server.py     # 网页版服务端（FastAPI + WebSocket）
├── web_llm.py        # LLM 流式调用
├── web_tts.py        # TTS 流式调用
├── web_asr.py        # 本地语音识别（FunASR）
├── config.py         # 共享配置，加载 .env
├── static/           # 网页前端（index.html + app.js）
├── requirements.txt  # Python 依赖
└── .env              # API 凭据（需自行创建，不入仓库）
```
