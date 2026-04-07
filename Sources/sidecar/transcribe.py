#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mlx-audio>=0.2.2",
#   "mlx-lm",
#   "numpy",
#   "soundfile",
#   "scipy",
#   "starlette",
#   "uvicorn",
# ]
# ///
"""Yuwp ASR — streaming speech-to-text with O(1) per-chunk latency.

Two modes:
  stdio   — JSON lines over stdin/stdout (Yuwp.app local dictation)
  serve   — HTTP server on localhost (external streaming STT clients)

Both modes share the same loaded model and streaming algorithm.

Stdio protocol:
  stdin  -> {"cmd": "start", "language": "en"}
  stdin  -> {"cmd": "audio", "pcm_b64": "..."}    # base64 Int16 PCM, 16kHz mono
  stdin  -> {"cmd": "stop"}
  stdin  -> {"cmd": "quit"}
  stdout <- {"type": "ready"}
  stdout <- {"type": "partial", "text": "..."}
  stdout <- {"type": "final", "text": "..."}
  stdout <- {"type": "error", "message": "..."}

HTTP endpoints (serve mode):
  POST   /v1/audio/transcriptions/stream       — create session
  POST   /v1/audio/transcriptions/stream/:id   — feed audio chunk (raw PCM)
  DELETE /v1/audio/transcriptions/stream/:id   — stop session, get final text
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
HOP_LENGTH = 160
DEFAULT_PORT = 9748

DEFAULT_CHUNK_SEC = 2.0
DEFAULT_ROLLBACK = 5
DEFAULT_UNFIXED_CHUNKS = 2
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_MAX_ENC_WINDOWS = 4
DEFAULT_MAX_PREFIX_TOKENS = 20

_SILENCE_RMS_THRESHOLD = 0.003

# Global inference lock — MLX Metal backend is not thread-safe.
# Concurrent matmul calls corrupt the command encoder (EXC_BAD_ACCESS in steel_matmul).
# Both stdio and HTTP paths must hold this lock during process_chunk / batch_retranscribe.
_inference_lock = threading.Lock()
_PAUSE_RMS_THRESHOLD = 0.020  # higher threshold for pause detection (real mic noise ~0.015)
_EOS_TOKENS = {151645, 151643}
_SESSION_TIMEOUT = 300  # seconds
_PAUSE_SILENCE_CHUNKS = 2  # consecutive quiet chunks before batch retranscribe (~2 * 2s = 4s)

# Language header tokens to strip from prefix/output
_ALL_LANG_TOKENS = {11528, 6364, 8453, 22574, 44923, 151704}

# Global model state (set by load_models)
_streaming_model: Any = None
_streaming_model_name: str = ""
_batch_model_wrapper: Any = None  # wrapper needed by generate_transcription
_batch_model_name: str = ""
_batch_retranscribe_enabled: bool = True


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[yuwp-asr] {msg}", file=sys.stderr, flush=True)


def send(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    chunk_sec: float = DEFAULT_CHUNK_SEC
    rollback: int = DEFAULT_ROLLBACK
    unfixed_chunks: int = DEFAULT_UNFIXED_CHUNKS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_enc_windows: int = DEFAULT_MAX_ENC_WINDOWS
    max_prefix_tokens: int = DEFAULT_MAX_PREFIX_TOKENS
    system_prompt: str | None = None
    model_name: str = ""


@dataclass
class StreamSession:
    model: Any
    config: StreamConfig

    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    enc_window_cache: list = field(default_factory=list)
    next_window_start: int = 0
    enc_window_samples: int = 0
    kv_cache: list = field(default=None)
    prev_prefill_embeds: mx.array = None
    raw_tokens: list = field(default_factory=list)
    chunk_idx: int = 0
    chunk_timings: list = field(default_factory=list)
    last_text: str = ""
    last_activity: float = field(default_factory=time.time)
    # Pause-triggered batch retranscription
    consecutive_silence: int = 0
    batch_done_for_pause: bool = False

    def __post_init__(self):
        n_window_infer = self.model.config.audio_config.n_window_infer
        self.enc_window_samples = n_window_infer * HOP_LENGTH
        self.kv_cache = self.model.make_cache()


# ---------------------------------------------------------------------------
# Encoder: incremental with window caching
# ---------------------------------------------------------------------------

def _preprocess_segment(model, audio_np: np.ndarray):
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import (
        _get_feat_extract_output_lengths,
    )

    audio_inputs = model._feature_extractor(
        audio_np,
        sampling_rate=SAMPLE_RATE,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    input_features = mx.array(audio_inputs["input_features"])
    feature_attention_mask = mx.array(audio_inputs["attention_mask"])
    audio_lengths = feature_attention_mask.sum(axis=-1).astype(mx.int32)
    aftercnn_lens = _get_feat_extract_output_lengths(audio_lengths)
    num_tokens = int(aftercnn_lens[0].item())
    return input_features, feature_attention_mask, num_tokens


def _encode_incremental(session: StreamSession) -> mx.array:
    model = session.model
    total_samples = len(session.audio_buffer)
    window_samples = session.enc_window_samples
    max_windows = session.config.max_enc_windows

    while session.next_window_start + window_samples <= total_samples:
        ws = session.next_window_start
        window_audio = session.audio_buffer[ws : ws + window_samples]
        features, mask, _ = _preprocess_segment(model, window_audio)
        enc_out = model.get_audio_features(features, mask)
        mx.eval(enc_out)
        session.enc_window_cache.append(enc_out)
        session.next_window_start += window_samples

    while len(session.enc_window_cache) > max_windows:
        session.enc_window_cache.pop(0)

    tail_enc = None
    if session.next_window_start < total_samples:
        tail_audio = session.audio_buffer[session.next_window_start:]
        if len(tail_audio) > 0:
            features, mask, _ = _preprocess_segment(model, tail_audio)
            tail_enc = model.get_audio_features(features, mask)
            mx.eval(tail_enc)

    parts = list(session.enc_window_cache)
    if tail_enc is not None:
        parts.append(tail_enc)

    if not parts:
        dim = model.config.audio_config.output_dim
        return mx.zeros((0, dim))

    return mx.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Embedding construction + delta prefill
# ---------------------------------------------------------------------------

def _build_input_embeds(
    model, enc_output: mx.array, prefix_tokens: list[int],
    system_prompt: str | None = None,
) -> mx.array:
    num_enc_tokens = enc_output.shape[0]
    input_ids = model._build_prompt(num_enc_tokens, language=None, system_prompt=system_prompt)
    inputs_embeds = model._build_inputs_embeds(input_ids, enc_output)

    if prefix_tokens:
        prefix_ids = mx.array([prefix_tokens])
        prefix_embeds = model.model.embed_tokens(prefix_ids)
        inputs_embeds = mx.concatenate([inputs_embeds, prefix_embeds], axis=1)

    return inputs_embeds


def _compute_reuse_length(prev_embeds: mx.array | None, new_embeds: mx.array) -> int:
    if prev_embeds is None:
        return 0
    cmp_len = min(prev_embeds.shape[1], new_embeds.shape[1])
    if cmp_len == 0:
        return 0
    prev_f32 = prev_embeds[0, :cmp_len].astype(mx.float32)
    new_f32 = new_embeds[0, :cmp_len].astype(mx.float32)
    diff = mx.abs(prev_f32 - new_f32).sum(axis=-1)
    mx.eval(diff)
    diff_np = np.array(diff)
    mismatches = np.where(diff_np > 1e-4)[0]
    return cmp_len if len(mismatches) == 0 else int(mismatches[0])


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def _decode_tokens(
    model, logits: mx.array, cache: list, max_tokens: int,
    *, repetition_window: int = 8, repetition_penalty: float = 1.3,
) -> list[int]:
    tokens: list[int] = []
    token = mx.argmax(logits[0, -1, :]).item()

    for _ in range(max_tokens):
        if token in _EOS_TOKENS:
            break
        tokens.append(token)
        if len(tokens) >= 4 and len(set(tokens[-4:])) == 1:
            break

        tok_embeds = model.model.embed_tokens(mx.array([[token]]))
        logits = model._forward_with_embeds(tok_embeds, cache=cache)

        recent = set(tokens[-repetition_window:])
        if recent:
            raw_logits = logits[0, -1, :]
            idx = mx.array(list(recent))
            vals = raw_logits[idx]
            penalized = mx.where(
                vals > 0, vals / repetition_penalty, vals * repetition_penalty
            )
            raw_logits[idx] = penalized
            mx.eval(raw_logits)
            token = mx.argmax(raw_logits).item()
        else:
            mx.eval(logits)
            token = mx.argmax(logits[0, -1, :]).item()

    return tokens


def _extract_text(model, raw_tokens: list[int]) -> str:
    text_tokens = [t for t in raw_tokens if t not in _ALL_LANG_TOKENS]
    text = model._tokenizer.decode(text_tokens, skip_special_tokens=True).strip()
    # Model outputs literal "None" for non-speech audio — treat as empty
    return "" if text == "None" else text


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

def _detect_repetition(text: str, min_period: int = 30) -> int | None:
    """Detect if text ends with a repeated phrase.

    Checks if the last `p` characters match the preceding `p` characters
    for increasing period lengths. When found, returns the character index
    to trim at (keeping the first occurrence of the repeated phrase).

    Returns None if no repetition detected.
    """
    n = len(text)
    if n < min_period * 2:
        return None

    for p in range(min_period, n // 2 + 1):
        if text[n - 2 * p : n - p] == text[n - p :]:
            # Found repeated period. Find earliest occurrence to keep.
            pattern = text[n - p :]
            first = text.find(pattern)
            if first < n - p:
                return first + p
    return None


def _trim_repetition(session: StreamSession) -> str | None:
    """Detect sentence-level repetition and trim raw_tokens.

    When the streaming decoder enters a loop (same phrase generated repeatedly),
    this function detects the repeated suffix, trims it from the text, and
    re-encodes the trimmed text back into raw_tokens so subsequent chunks
    don't continue the loop.

    Returns trimmed text if repetition was found, None otherwise.
    """
    text = _extract_text(session.model, session.raw_tokens)
    trim_at = _detect_repetition(text)
    if trim_at is None:
        return None

    trimmed = text[:trim_at].rstrip()
    # Re-tokenize the trimmed text so raw_tokens stays in sync.
    # This may differ slightly from the original decode tokens, but
    # the prefix mechanism will re-anchor the decoder on the next chunk.
    new_tokens = session.model._tokenizer.encode(trimmed)
    new_tokens = [t for t in new_tokens if t not in _ALL_LANG_TOKENS]
    period = len(text) - trim_at
    log(f"Repetition detected (period={period} chars, trimmed {len(text)}→{len(trimmed)} chars)")
    session.raw_tokens = new_tokens
    return trimmed


# ---------------------------------------------------------------------------
# Pause-triggered batch retranscription
# ---------------------------------------------------------------------------

def _batch_retranscribe(session: StreamSession) -> str | None:
    """Run full batch transcription on accumulated audio.

    Uses the global _batch_model_wrapper for a single-pass decode
    over the entire audio buffer. Returns corrected text, or None if
    disabled/unavailable. Re-tokenizes raw_tokens so streaming can
    continue from the corrected state.
    """
    if not _batch_retranscribe_enabled:
        return None
    if _batch_model_wrapper is None:
        return None  # batch model not loaded yet (background load in progress)
    if len(session.audio_buffer) < SAMPLE_RATE:  # <1s, skip
        return None

    try:
        from mlx_audio.stt.generate import generate_transcription
        import tempfile, soundfile as sf

        # Write accumulated audio to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(f, session.audio_buffer, SAMPLE_RATE)

        t0 = time.time()
        result = generate_transcription(_batch_model_wrapper, tmp_path)
        batch_ms = (time.time() - t0) * 1000

        import os
        os.unlink(tmp_path)

        text = result.text.strip()
        audio_s = len(session.audio_buffer) / SAMPLE_RATE
        log(f"Batch retranscribe: {audio_s:.1f}s audio in {batch_ms:.0f}ms → {len(text)} chars")

        # Re-tokenize so streaming prefix tokens continue from batch result
        if text:
            new_tokens = session.model._tokenizer.encode(text)
            new_tokens = [t for t in new_tokens if t not in _ALL_LANG_TOKENS]
            session.raw_tokens = new_tokens

        return text
    except Exception as e:
        log(f"Batch retranscribe error: {e}")
        return None


# ---------------------------------------------------------------------------
# Core: process one chunk
# ---------------------------------------------------------------------------

def process_chunk(session: StreamSession, audio_chunk: np.ndarray) -> dict:
    """Process one audio chunk. Returns dict with text and timing."""
    chunk_t0 = time.time()
    cfg = session.config

    # Track audio level for both silence skip and pause detection
    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

    # Pause detection uses a higher threshold than encoding skip
    if rms < _PAUSE_RMS_THRESHOLD:
        session.consecutive_silence += 1
    else:
        session.consecutive_silence = 0
        session.batch_done_for_pause = False

    # Pause detected: batch retranscribe synchronously.
    # This runs on the main thread (MLX is not thread-safe) but only during
    # silence, so it doesn't block speech processing. The ~1s batch runs
    # while the user is pausing — the next feed arrives ~2s later.
    if (session.consecutive_silence >= _PAUSE_SILENCE_CHUNKS
            and not session.batch_done_for_pause
            and session.raw_tokens):
        session.batch_done_for_pause = True
        batch_text = _batch_retranscribe(session)
        if batch_text is not None:
            session.last_text = batch_text
            session.chunk_idx += 1
            session.last_activity = time.time()
            return {
                "text": batch_text,
                "is_partial": True,
                "batch_corrected": True,
                "total_ms": (time.time() - chunk_t0) * 1000,
            }

    if rms < _SILENCE_RMS_THRESHOLD:
        session.chunk_idx += 1
        session.last_activity = time.time()
        text = _extract_text(session.model, session.raw_tokens) if session.raw_tokens else ""
        session.last_text = text
        return {"text": text, "is_partial": True, "total_ms": (time.time() - chunk_t0) * 1000}

    # Append audio
    session.audio_buffer = np.concatenate([session.audio_buffer, audio_chunk])

    # Encode incrementally
    t0 = time.time()
    enc_output = _encode_incremental(session)
    encode_ms = (time.time() - t0) * 1000

    if enc_output.shape[0] == 0:
        session.chunk_idx += 1
        return {"text": "", "is_partial": True, "total_ms": (time.time() - chunk_t0) * 1000}

    # Build prefix tokens (rollback)
    prefix_tokens = []
    if session.chunk_idx >= cfg.unfixed_chunks and session.raw_tokens:
        n_prefix = len(session.raw_tokens) - cfg.rollback
        if n_prefix < 0:
            n_prefix = 0
        prefix_tokens = session.raw_tokens[:n_prefix]
        prefix_tokens = [t for t in prefix_tokens if t not in _ALL_LANG_TOKENS]
        if len(prefix_tokens) > cfg.max_prefix_tokens:
            prefix_tokens = prefix_tokens[-cfg.max_prefix_tokens:]

    # Build input embeddings
    input_embeds = _build_input_embeds(
        session.model, enc_output, prefix_tokens,
        system_prompt=cfg.system_prompt,
    )
    mx.eval(input_embeds)

    # Delta prefill
    t0 = time.time()
    reuse_len = _compute_reuse_length(session.prev_prefill_embeds, input_embeds)
    total_prefill_len = input_embeds.shape[1]

    for c in session.kv_cache:
        c.offset = reuse_len

    prefill_len = total_prefill_len - 1
    delta_len = prefill_len - reuse_len

    if delta_len > 0:
        delta_embeds = input_embeds[:, reuse_len:prefill_len, :]
        logits = session.model._forward_with_embeds(delta_embeds, cache=session.kv_cache)
        mx.eval(logits)

    last_embed = input_embeds[:, prefill_len : prefill_len + 1, :]
    logits = session.model._forward_with_embeds(last_embed, cache=session.kv_cache)
    mx.eval(logits)

    prefill_ms = (time.time() - t0) * 1000
    reuse_pct = reuse_len / max(total_prefill_len, 1) * 100
    session.prev_prefill_embeds = input_embeds[:, :prefill_len, :]

    # Decode
    t0 = time.time()
    new_tokens = _decode_tokens(session.model, logits, session.kv_cache, cfg.max_new_tokens)
    decode_ms = (time.time() - t0) * 1000

    # Update raw token history
    if new_tokens:
        uncapped_prefix = []
        if session.chunk_idx >= cfg.unfixed_chunks and session.raw_tokens:
            n = len(session.raw_tokens) - cfg.rollback
            if n > 0:
                uncapped_prefix = session.raw_tokens[:n]
        session.raw_tokens = uncapped_prefix + new_tokens

    text = _extract_text(session.model, session.raw_tokens)

    # Detect and trim sentence-level repetition loops
    trimmed = _trim_repetition(session)
    if trimmed is not None:
        text = trimmed

    session.last_text = text
    session.last_activity = time.time()

    total_ms = (time.time() - chunk_t0) * 1000
    session.chunk_timings.append(total_ms)
    session.chunk_idx += 1

    return {
        "text": text,
        "is_partial": True,
        "encode_ms": round(encode_ms, 1),
        "prefill_ms": round(prefill_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(total_ms, 1),
        "reuse_pct": round(reuse_pct, 1),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    streaming_name: str,
    batch_name: str | None = None,
    batch_enabled: bool = True,
) -> Any:
    """Load streaming model (returned) and optionally a batch model.

    If batch is enabled and uses the same model, the wrapper is shared (zero
    extra memory). If batch uses a different model, it loads in a background
    thread so streaming is available immediately.

    Returns the unwrapped streaming model for use in StreamSession.
    """
    global _streaming_model, _streaming_model_name
    global _batch_model_wrapper, _batch_model_name, _batch_retranscribe_enabled
    from mlx_audio.stt import load_model as _load

    _batch_retranscribe_enabled = batch_enabled

    # Load streaming model (always needed)
    log(f"Loading streaming model: {streaming_name}")
    t0 = time.time()
    streaming_wrapper = _load(streaming_name)
    streaming_model = (
        streaming_wrapper._model
        if hasattr(streaming_wrapper, "_model")
        else streaming_wrapper
    )
    _streaming_model = streaming_model
    _streaming_model_name = streaming_name
    log(f"Streaming model loaded in {time.time() - t0:.1f}s")

    # Batch model
    if batch_enabled:
        batch_name = batch_name or streaming_name
        _batch_model_name = batch_name

        if batch_name == streaming_name:
            _batch_model_wrapper = streaming_wrapper
            log(f"Batch model: reusing streaming model ({batch_name})")
        else:
            # Load in background so streaming is available immediately
            def _load_batch_bg() -> None:
                global _batch_model_wrapper
                log(f"Loading batch model: {batch_name}")
                bt0 = time.time()
                _batch_model_wrapper = _load(batch_name)
                log(f"Batch model loaded in {time.time() - bt0:.1f}s")

            threading.Thread(
                target=_load_batch_bg, daemon=True, name="batch-model-load"
            ).start()
    else:
        log("Batch retranscription disabled")

    return streaming_model


# ---------------------------------------------------------------------------
# HTTP server (serve mode)
# ---------------------------------------------------------------------------

class SessionManager:
    """Thread-safe session store for HTTP streaming sessions."""

    def __init__(self, model):
        self.model = model
        self.sessions: dict[str, StreamSession] = {}
        self.pending_audio: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="session-cleanup"
        )
        self._cleanup_thread.start()

    def create(self, stream_config: dict | None = None) -> str:
        sid = uuid.uuid4().hex[:12]
        cfg = StreamConfig(model_name=_streaming_model_name)
        if stream_config and isinstance(stream_config, dict):
            if "system_prompt" in stream_config:
                cfg.system_prompt = stream_config["system_prompt"]
        with self.lock:
            self.sessions[sid] = StreamSession(model=self.model, config=cfg)
            self.pending_audio[sid] = np.array([], dtype=np.float32)
        log(f"HTTP session created: {sid}")
        return sid

    def feed(self, sid: str, pcm_bytes: bytes) -> dict | None:
        """Feed raw s16le 16kHz mono PCM. Returns {text, batch_corrected?} or None."""
        with self.lock:
            session = self.sessions.get(sid)
            pending = self.pending_audio.get(sid)
            if session is None or pending is None:
                return None

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        pending = np.concatenate([pending, samples])

        batch_corrected = False
        chunk_samples = int(session.config.chunk_sec * SAMPLE_RATE)
        while len(pending) >= chunk_samples:
            chunk = pending[:chunk_samples]
            pending = pending[chunk_samples:]
            with _inference_lock:
                try:
                    result = process_chunk(session, chunk)
                    if result.get("batch_corrected"):
                        batch_corrected = True
                except Exception as e:
                    log(f"HTTP feed error ({sid}): {e}")

        with self.lock:
            if sid in self.pending_audio:
                self.pending_audio[sid] = pending

        resp = {"text": session.last_text}
        if batch_corrected:
            resp["batch_corrected"] = True
        return resp

    def stop(self, sid: str) -> str | None:
        """Stop session, process remaining audio, return final text."""
        with self.lock:
            session = self.sessions.pop(sid, None)
            pending = self.pending_audio.pop(sid, None)
            if session is None:
                return None

        # Process leftover audio + final batch (under inference lock)
        with _inference_lock:
            if pending is not None and len(pending) > 0:
                try:
                    process_chunk(session, pending)
                except Exception as e:
                    log(f"HTTP final chunk error ({sid}): {e}")

            streaming_text = _extract_text(session.model, session.raw_tokens) if session.raw_tokens else ""

            # Final batch retranscribe for best quality
            batch_text = _batch_retranscribe(session) if session.raw_tokens else None
            text = batch_text if batch_text else streaming_text

        total_audio = len(session.audio_buffer) / SAMPLE_RATE
        avg_ms = sum(session.chunk_timings) / max(len(session.chunk_timings), 1)

        log(f"HTTP session stopped ({sid}): {total_audio:.1f}s audio, {session.chunk_idx} chunks, {avg_ms:.0f}ms avg")
        return text

    def delete(self, sid: str) -> bool:
        """Remove session without processing remaining audio."""
        with self.lock:
            removed = self.sessions.pop(sid, None) is not None
            self.pending_audio.pop(sid, None)
        return removed

    def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            time.sleep(30)
            now = time.time()
            with self.lock:
                expired = [
                    sid for sid, s in self.sessions.items()
                    if now - s.last_activity > _SESSION_TIMEOUT
                ]
                for sid in expired:
                    self.sessions.pop(sid, None)
                    self.pending_audio.pop(sid, None)
                    log(f"Session {sid} expired (timeout)")


def make_app(session_mgr: SessionManager):
    """Build Starlette ASGI app with the streaming session endpoints."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def create_session(request: Request) -> JSONResponse:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        stream_config = body.get("stream_config")
        sid = session_mgr.create(stream_config)
        return JSONResponse({"session_id": sid})

    async def feed_audio(request: Request) -> JSONResponse:
        sid = request.path_params["session_id"]
        pcm = await request.body()
        result = session_mgr.feed(sid, pcm)
        if result is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse(result)

    async def stop_session(request: Request) -> JSONResponse:
        sid = request.path_params["session_id"]
        text = session_mgr.stop(sid)
        if text is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"text": text})

    async def info(request: Request) -> JSONResponse:
        resp: dict[str, Any] = {
            "streaming_model": _streaming_model_name,
            "sample_rate": SAMPLE_RATE,
            "chunk_sec": DEFAULT_CHUNK_SEC,
            "status": "ready",
            "batch_retranscribe": _batch_retranscribe_enabled,
        }
        if _batch_retranscribe_enabled:
            resp["batch_model"] = _batch_model_name
            resp["batch_model_loaded"] = _batch_model_wrapper is not None
        return JSONResponse(resp)

    return Starlette(routes=[
        Route("/v1/info", info, methods=["GET"]),
        Route("/v1/audio/transcriptions/stream", create_session, methods=["POST"]),
        Route("/v1/audio/transcriptions/stream/{session_id}", feed_audio, methods=["POST"]),
        Route("/v1/audio/transcriptions/stream/{session_id}", stop_session, methods=["DELETE"]),
    ])


def start_http_server(model, host: str, port: int) -> None:
    """Start HTTP server in a background thread."""
    import uvicorn

    session_mgr = SessionManager(model)
    app = make_app(session_mgr)

    config = uvicorn.Config(
        app, host=host, port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True, name="yuwp-http")
    thread.start()
    log(f"HTTP server listening on {host}:{port}")


# ---------------------------------------------------------------------------
# Main stdio loop
# ---------------------------------------------------------------------------

def run_stdio(model) -> None:
    """Run the stdio JSON-lines protocol for Yuwp.app."""
    session: StreamSession | None = None
    pending_audio = np.array([], dtype=np.float32)
    chunk_samples = int(DEFAULT_CHUNK_SEC * SAMPLE_RATE)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            send({"type": "error", "message": f"Invalid JSON: {e}"})
            continue

        cmd = msg.get("cmd")

        if cmd == "start":
            cfg = StreamConfig(model_name=_streaming_model_name)
            session = StreamSession(model=model, config=cfg)
            pending_audio = np.array([], dtype=np.float32)
            log("Session started")

        elif cmd == "audio" and session is not None:
            pcm_bytes = base64.b64decode(msg["pcm_b64"])
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            pending_audio = np.concatenate([pending_audio, samples])

            while len(pending_audio) >= chunk_samples:
                chunk = pending_audio[:chunk_samples]
                pending_audio = pending_audio[chunk_samples:]
                with _inference_lock:
                    try:
                        result = process_chunk(session, chunk)
                        send({"type": "partial", "text": result["text"]})
                    except Exception as e:
                        log(f"Chunk error: {e}")
                        send({"type": "error", "message": str(e)})

        elif cmd == "stop" and session is not None:
            with _inference_lock:
                if len(pending_audio) > 0:
                    try:
                        process_chunk(session, pending_audio)
                    except Exception as e:
                        log(f"Final chunk error: {e}")

                # Streaming text
                streaming_text = _extract_text(model, session.raw_tokens) if session.raw_tokens else ""

                # Final batch retranscribe for best quality (same as HTTP stop)
                batch_text = _batch_retranscribe(session) if session.raw_tokens else None
                text = batch_text if batch_text else streaming_text

            total_audio = len(session.audio_buffer) / SAMPLE_RATE
            avg_ms = sum(session.chunk_timings) / max(len(session.chunk_timings), 1)
            log(f"Session stopped: {total_audio:.1f}s audio, {session.chunk_idx} chunks, {avg_ms:.0f}ms avg")
            send({"type": "final", "text": text})
            session = None
            pending_audio = np.array([], dtype=np.float32)

        elif cmd == "quit":
            log("Shutting down")
            break

        else:
            send({"type": "error", "message": f"Unknown command: {cmd}"})

    log("Exiting")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Yuwp ASR — streaming speech-to-text")
    parser.add_argument("model", nargs="?", default="mlx-community/Qwen3-ASR-0.6B-4bit",
                        help="Streaming model name or path (default: mlx-community/Qwen3-ASR-0.6B-4bit)")
    parser.add_argument("--batch-model", default=None,
                        help="Batch retranscription model (default: same as streaming model)")
    parser.add_argument("--no-batch-retranscribe", action="store_true",
                        help="Disable batch retranscription entirely")
    parser.add_argument("--serve", action="store_true",
                        help="Start HTTP server alongside stdio (for external clients)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP server bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP server port (default: {DEFAULT_PORT})")
    parser.add_argument("--serve-only", action="store_true",
                        help="Run HTTP server only, no stdio (for standalone ASR server)")
    args = parser.parse_args()

    batch_enabled = not args.no_batch_retranscribe
    model = load_models(args.model, args.batch_model, batch_enabled)

    if args.serve or args.serve_only:
        start_http_server(model, args.host, args.port)

    if args.serve_only:
        log("Running in serve-only mode (no stdio). Press Ctrl+C to exit.")
        send({"type": "ready"})
        try:
            threading.Event().wait()  # Block forever
        except KeyboardInterrupt:
            log("Shutting down")
    else:
        send({"type": "ready"})
        run_stdio(model)


if __name__ == "__main__":
    main()
