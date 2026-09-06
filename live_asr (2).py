"""
Live Correct - real-time recitation feedback.

WHY THIS FILE EXISTS AND HOW IT WORKS
---------------------------------------
The existing "Listen & Correct" flow is record -> submit -> transcribe the
whole clip -> grade once. This module adds a second path: keep listening
continuously, re-transcribe what's been said so far every couple of
seconds, and re-run grading on that growing transcript - so the student
sees corrections while still reciting, not after they stop.

Three real constraints shaped every decision below, and it's worth
understanding them before touching this file:

1. Streamlit has no native "keep a background process running while the
   UI stays interactive" model - it reruns the whole script on each
   interaction. Continuous mic capture needs `streamlit-webrtc`, which
   runs its own audio thread outside Streamlit's rerun cycle and hands
   frames back through a thread-safe queue. This module ONLY reads that
   queue from the main Streamlit thread (on each autorefresh tick)  - 
   never mutates st.session_state from inside a WebRTC callback, which
   is unsafe and a common source of silent bugs in streamlit-webrtc code
   found online.

2. The existing ASR path (`transformers` pipeline running the Quranic
   Whisper LoRA fine-tune on CPU) is too slow to re-run every 2 seconds  - 
   a single pass can itself take several seconds. This module uses
   `faster-whisper` (CTranslate2 backend, int8-quantized) instead, which
   is meaningfully faster on CPU. That means LIVE MODE USES A DIFFERENT,
   SEPARATE MODEL FILE from the batch mode until that model has been
   converted - see `convert_model_for_live.py` in this same folder,
   which must be run once, offline, before this module has a real model
   to load. Until that conversion is done, this module fails loudly with
   a clear setup instruction rather than silently degrading.

3. Rather than build new incremental-diffing logic, every tick simply
   re-transcribes the FULL buffer captured so far and re-runs the
   existing, already-tested `grade_continuous_recitation()` against it.
   This is deliberately the simple option: it reuses tested alignment
   logic instead of writing a new one, at the cost of doing slightly
   more work each tick as the buffer grows. For a typical Hifz test
   segment (a few ayahs, well under a couple of minutes) this is a fair
   trade. `MAX_LIVE_BUFFER_SECONDS` bounds it so a very long session
   can't degrade into multi-second tick times.

WHAT I COULD NOT TEST HERE
---------------------------
This sandbox has no microphone, no browser, and no way to run a live
WebRTC session. The audio-frame conversion logic below follows
streamlit-webrtc's documented frame format, but you need to be the one
who confirms it actually works end-to-end locally - see the manual test
checklist at the bottom of DESKTOP_BUILD.md-style docs in the setup
guide I'm giving you alongside this file. Treat this as a working first
draft to debug against real hardware, not a guaranteed-correct feature.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np
import streamlit as st

SAMPLE_RATE = 16000  # required input rate for Whisper-family models
MAX_LIVE_BUFFER_SECONDS = 120  # bound per-tick re-transcription cost
TICK_MIN_NEW_SECONDS = 1.0     # don't bother re-running ASR on tiny slivers of new audio

LIVE_ASR_MODEL_PATH = os.environ.get(
    "LIVE_ASR_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_model_ct2"),
)

# If you converted the model somewhere other than this machine (e.g. Google
# Colab, since the conversion itself needs more RAM/disk than a typical dev
# laptop or the Streamlit Cloud build environment has) and pushed the result
# to a Hugging Face Hub model repo, set this instead of shipping the model
# in the repo. See colab_convert_and_upload.py.
LIVE_ASR_HF_REPO = os.environ.get("LIVE_ASR_HF_REPO", "")


class LiveAudioBuffer:
    """Thread-safe, growing mono float32 @ 16kHz audio buffer.

    Frames arrive on streamlit-webrtc's internal thread via the audio
    receiver queue (polled from the main thread - see poll_webrtc_frames
    below), get resampled/converted here, and are appended under a lock.
    Reads (by the main thread, once per autorefresh tick) take a copy
    under the same lock so the writer never mutates memory a reader is
    part-way through using.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples = np.zeros(0, dtype=np.float32)
        self._last_read_len = 0

    def append(self, samples: np.ndarray) -> None:
        with self._lock:
            self._samples = np.concatenate([self._samples, samples])
            # Bound memory/compute: keep only the most recent N seconds.
            max_len = MAX_LIVE_BUFFER_SECONDS * SAMPLE_RATE
            if len(self._samples) > max_len:
                trimmed = len(self._samples) - max_len
                self._samples = self._samples[trimmed:]
                self._last_read_len = max(0, self._last_read_len - trimmed)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self._samples.copy()

    def new_seconds_available(self) -> float:
        with self._lock:
            return (len(self._samples) - self._last_read_len) / SAMPLE_RATE

    def mark_read(self) -> None:
        with self._lock:
            self._last_read_len = len(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples = np.zeros(0, dtype=np.float32)
            self._last_read_len = 0

    def total_seconds(self) -> float:
        with self._lock:
            return len(self._samples) / SAMPLE_RATE


def _resample_frame(frame_array: np.ndarray, in_rate: int) -> np.ndarray:
    """Converts one WebRTC audio frame to mono float32 @ 16kHz."""
    if frame_array.ndim > 1:
        frame_array = frame_array.mean(axis=0) if frame_array.shape[0] < frame_array.shape[1] else frame_array.mean(axis=1)
    frame_array = frame_array.astype(np.float32)
    # int16 PCM from most WebRTC sources needs normalizing to [-1, 1]
    if np.max(np.abs(frame_array)) > 1.5:
        frame_array = frame_array / 32768.0
    if in_rate != SAMPLE_RATE:
        try:
            import librosa
            frame_array = librosa.resample(frame_array, orig_sr=in_rate, target_sr=SAMPLE_RATE)
        except ImportError:
            ratio = SAMPLE_RATE / in_rate
            new_len = max(1, int(len(frame_array) * ratio))
            frame_array = np.interp(
                np.linspace(0, len(frame_array) - 1, new_len),
                np.arange(len(frame_array)),
                frame_array,
            )
    return frame_array


def poll_webrtc_frames(webrtc_ctx, buffer: LiveAudioBuffer, timeout: float = 1.0) -> int:
    """
    Call this once per autorefresh tick from the MAIN Streamlit thread
    (not from inside a callback). Drains whatever audio frames
    streamlit-webrtc has queued up since the last poll and appends them
    to `buffer`. Returns how many frames were drained (0 is normal if the
    user is silent or between ticks).
    """
    if webrtc_ctx is None or not webrtc_ctx.state.playing or webrtc_ctx.audio_receiver is None:
        return 0

    try:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
    except queue.Empty:
        return 0

    for frame in audio_frames:
        arr = frame.to_ndarray()
        converted = _resample_frame(arr, frame.sample_rate)
        buffer.append(converted)

    return len(audio_frames)


@st.cache_resource(show_spinner="Loading live recitation model \u2026")
def load_live_asr_model():
    """
    Loads the CTranslate2-converted model for fast CPU inference.

    Deliberately fails with an actionable message rather than silently
    falling back to the slow batch model - a live feature that's
    secretly running at batch speed is worse than one that clearly says
    "not set up yet", because the lag would look like a bug, not a
    missing setup step.

    Resolution order for the model files:
    1. LIVE_ASR_MODEL_PATH if it already exists locally (fastest - no
       network needed; this is the case after a Docker build that baked
       the model in, or on a dev machine that ran the conversion itself).
    2. Otherwise, if LIVE_ASR_HF_REPO is set, download the converted
       model from that Hugging Face Hub repo into LIVE_ASR_MODEL_PATH.
       This is the path for Streamlit Community Cloud, where the
       conversion itself can't run (not enough RAM/disk/build time) but
       downloading a few hundred MB of already-converted weights at
       startup is fine.
    3. Otherwise, fail loudly with setup instructions.
    """
    if not os.path.isdir(LIVE_ASR_MODEL_PATH) and LIVE_ASR_HF_REPO:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=LIVE_ASR_HF_REPO,
            local_dir=LIVE_ASR_MODEL_PATH,
            token=os.environ.get("HF_TOKEN") or None,  # only needed for a private repo
        )

    if not os.path.isdir(LIVE_ASR_MODEL_PATH):
        raise RuntimeError(
            f"Live ASR model not found at '{LIVE_ASR_MODEL_PATH}', and no "
            "LIVE_ASR_HF_REPO is set to download it from. Live Correct "
            "needs a CTranslate2-converted copy of the recitation model "
            "for fast-enough CPU inference. If you have a machine with "
            "enough RAM/disk, run `python convert_model_for_live.py` "
            "once (see LIVE_CORRECT_SETUP.md). If not, run "
            "colab_convert_and_upload.py on Google Colab (free) and set "
            "LIVE_ASR_HF_REPO to where it uploads the result, then "
            "restart the app."
        )
    from faster_whisper import WhisperModel
    return WhisperModel(LIVE_ASR_MODEL_PATH, device="cpu", compute_type="int8")


def transcribe_buffer(model, audio: np.ndarray) -> str:
    """Runs faster-whisper on the full buffer captured so far and returns
    plain concatenated text - same shape of output as the batch
    transcribe_audio(), so it can be fed straight into the existing,
    unmodified clean_asr_output() / grade_continuous_recitation()."""
    if len(audio) < SAMPLE_RATE * 0.3:  # less than ~0.3s - not worth a pass
        return ""
    segments, _info = model.transcribe(
        audio,
        language="ar",
        task="transcribe",
        vad_filter=True,  # skips silence, keeps ticks fast as pauses happen
        beam_size=1,      # greedy - favors speed over the marginal accuracy
                          # a live in-progress reading doesn't benefit from
    )
    return " ".join(seg.text for seg in segments).strip()
