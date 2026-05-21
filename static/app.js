// Browser client: WebSocket, mic capture (MediaRecorder -> WebM/Opus blob),
// PCM playback scheduling (Web Audio API), barge-in.

const HEADER_LEN = 16;
const MAGIC0 = 0xa1, MAGIC1 = 0xb2;
const SAMPLE_RATE_TTS = 32000;
const PLAYBACK_LEAD_SEC = 0.2;

const chatEl = document.getElementById("chat");
const statusEl = document.getElementById("status");
const micBtn = document.getElementById("mic");
const textInput = document.getElementById("text");
const sendBtn = document.getElementById("send");

let ws = null;
let wsBackoff = 500;
let currentTurnId = 0;       // latest server-acknowledged turn (from llm_delta/tts headers)
let liveTurnId = 0;          // turn whose audio we're actively playing
const liveSources = new Set();
let playbackCtx = null;
let nextStartTime = null;

let micStream = null;
let mediaRecorder = null;
let recordedChunks = [];      // Blob[] from MediaRecorder dataavailable
let recording = false;
let currentBotBubble = null;  // {textNode, thinkingNode, turn_id}

function setStatus(s) { statusEl.textContent = s; }

function addUserBubble(text) {
  const row = document.createElement("div");
  row.className = "row user";
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function newBotBubble(turnId) {
  const row = document.createElement("div");
  row.className = "row bot";
  const wrap = document.createElement("div");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const textNode = document.createTextNode("");
  bubble.appendChild(textNode);
  const thinking = document.createElement("div");
  thinking.className = "thinking";
  wrap.appendChild(bubble);
  wrap.appendChild(thinking);
  row.appendChild(wrap);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  currentBotBubble = { textNode, thinking, turn_id: turnId, bubble };
  return currentBotBubble;
}

function ensureBotBubble(turnId) {
  if (!currentBotBubble || currentBotBubble.turn_id !== turnId) {
    return newBotBubble(turnId);
  }
  return currentBotBubble;
}

function ensurePlaybackCtx() {
  if (!playbackCtx) {
    try {
      playbackCtx = new AudioContext({ sampleRate: SAMPLE_RATE_TTS });
    } catch (_e) {
      playbackCtx = new AudioContext();
    }
  }
  if (playbackCtx.state === "suspended") playbackCtx.resume();
  return playbackCtx;
}

function stopAllPlayback() {
  liveSources.forEach((s) => {
    try { s.stop(); } catch (_e) {}
  });
  liveSources.clear();
  nextStartTime = null;
}

function schedulePcm(pcmInt16, ctxSr) {
  const ctx = ensurePlaybackCtx();
  // Convert int16 -> float32 [-1, 1).
  const frames = pcmInt16.length;
  if (frames === 0) return;
  const buffer = ctx.createBuffer(1, frames, ctxSr);
  const ch = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) ch[i] = pcmInt16[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  if (nextStartTime === null || nextStartTime < ctx.currentTime) {
    nextStartTime = ctx.currentTime + PLAYBACK_LEAD_SEC;
  }
  src.start(nextStartTime);
  nextStartTime += buffer.duration;
  liveSources.add(src);
  src.onended = () => liveSources.delete(src);
}

function handleBinaryFrame(buf) {
  const dv = new DataView(buf);
  if (dv.getUint8(0) !== MAGIC0 || dv.getUint8(1) !== MAGIC1) return;
  // version dv.getUint8(2)
  const turnId = dv.getUint32(3, true);
  const seq = dv.getUint32(7, true);
  const flags = dv.getUint8(11);
  if (turnId !== currentTurnId) return; // stale frame from cancelled turn
  if (liveTurnId !== turnId) {
    // new turn audio: ensure no leftover queue from a previous turn
    stopAllPlayback();
    liveTurnId = turnId;
  }
  const pcmBytes = buf.byteLength - HEADER_LEN;
  if (pcmBytes <= 0) {
    // final-only marker
    return;
  }
  const pcm = new Int16Array(buf, HEADER_LEN, pcmBytes / 2);
  schedulePcm(pcm, SAMPLE_RATE_TTS);
  if (flags & 1) {
    // sentence final flag (informational; no action needed beyond scheduling)
  }
}

function handleJsonMessage(obj) {
  const t = obj.type;
  if (t === "asr_result") {
    currentTurnId = obj.turn_id;
    if (obj.text) addUserBubble(obj.text);
    ensureBotBubble(obj.turn_id);
    setStatus("AI 思考中…");
  } else if (t === "thinking_delta") {
    currentTurnId = obj.turn_id;
    const b = ensureBotBubble(obj.turn_id);
    b.thinking.textContent += obj.text;
    chatEl.scrollTop = chatEl.scrollHeight;
  } else if (t === "llm_delta") {
    currentTurnId = obj.turn_id;
    const b = ensureBotBubble(obj.turn_id);
    b.textNode.nodeValue += obj.text;
    setStatus("AI 回复中…");
    chatEl.scrollTop = chatEl.scrollHeight;
  } else if (t === "llm_done") {
    setStatus("就绪");
  } else if (t === "tts_sentence_done") {
    // informational
  } else if (t === "error") {
    setStatus(`[${obj.where}] ${obj.message || "错误"}`);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/ws`;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    wsBackoff = 500;
    setStatus("已连接，就绪");
  };
  ws.onclose = () => {
    setStatus("已断开，正在重连…");
    setTimeout(connect, wsBackoff);
    wsBackoff = Math.min(wsBackoff * 2, 8000);
  };
  ws.onerror = () => {
    try { ws.close(); } catch (_e) {}
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      try { handleJsonMessage(JSON.parse(ev.data)); } catch (_e) {}
    } else {
      handleBinaryFrame(ev.data);
    }
  };
}

function wsSend(obj) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(obj));
}

function wsSendBinary(buf) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(buf);
}

// ---- text input ----
function sendText() {
  const text = textInput.value.trim();
  if (!text) return;
  // Barge-in: clicking send while AI talks should interrupt too.
  stopAllPlayback();
  addUserBubble(text);
  wsSend({ type: "text_input", text });
  textInput.value = "";
  setStatus("AI 思考中…");
}

sendBtn.addEventListener("click", sendText);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendText();
  }
});

// ---- mic capture (MediaRecorder) ----
function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
  ];
  for (const t of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

async function startRecording() {
  // Barge-in: any AI audio still playing must stop immediately.
  stopAllPlayback();
  wsSend({ type: "audio_start" });

  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const mime = pickMimeType();
  const opts = mime ? { mimeType: mime } : undefined;
  mediaRecorder = new MediaRecorder(micStream, opts);
  console.log("[mic] MediaRecorder mimeType =", mediaRecorder.mimeType);
  recordedChunks = [];
  mediaRecorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size > 0) recordedChunks.push(ev.data);
  };
  // Capture state we need at stop time (closures avoid races).
  let resolveStopped;
  const stopped = new Promise((res) => (resolveStopped = res));
  mediaRecorder.onstop = () => resolveStopped();
  mediaRecorder.start();
  // Stash the promise on the recorder so stopRecording can await it.
  mediaRecorder._stoppedPromise = stopped;
  recording = true;
  micBtn.classList.add("recording");
  micBtn.textContent = "⏹ 停止并发送";
  setStatus("录音中…");
}

async function stopRecording() {
  recording = false;
  micBtn.classList.remove("recording");
  micBtn.textContent = "🎙 按下说话";

  if (!mediaRecorder) {
    setStatus("未捕获到音频");
    return;
  }
  const stoppedPromise = mediaRecorder._stoppedPromise;
  try {
    mediaRecorder.requestData(); // flush any buffered data
  } catch (_e) {}
  if (mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  await stoppedPromise;

  try {
    if (micStream) micStream.getTracks().forEach((t) => t.stop());
  } catch (_e) {}
  micStream = null;
  mediaRecorder = null;

  if (recordedChunks.length === 0) {
    setStatus("未捕获到音频");
    return;
  }
  const blob = new Blob(recordedChunks, { type: recordedChunks[0].type || "audio/webm" });
  recordedChunks = [];
  setStatus("识别中…");
  const buf = await blob.arrayBuffer();
  console.log("[mic] sending", buf.byteLength, "bytes,", blob.type);
  wsSendBinary(buf);
  wsSend({ type: "audio_end" });
}

micBtn.addEventListener("click", async () => {
  try {
    if (!recording) await startRecording();
    else await stopRecording();
  } catch (e) {
    setStatus(`麦克风错误：${e.message || e}`);
    recording = false;
    micBtn.classList.remove("recording");
    micBtn.textContent = "🎙 按下说话";
  }
});

connect();
