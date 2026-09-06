# Live Correct — setup guide

Live Correct adds real-time feedback while reciting, instead of waiting
until you stop. This is new, beta, and genuinely more complex than the
rest of the app — read the whole thing before flipping it on for
students.

## Why this needs a setup step at all

The existing ASR model runs through the `transformers` library, which is
too slow on CPU to re-transcribe every ~2 seconds. Live Correct uses a
second, faster copy of the same model via `faster-whisper`. Getting that
fast copy requires a one-time conversion — this isn't optional, and the
app will refuse to enable Live Correct until it's done (on purpose —
better a clear error than silently falling back to the slow model and
just feeling "broken").

## 1. Install the extra dependencies

```bash
pip install -r requirements_live.txt
pip install peft ctranslate2 "transformers[torch]"   # only needed for the conversion step below
```

## 2. Run the one-time model conversion

```bash
python convert_model_for_live.py
```

This downloads the base Whisper model, merges in the Quran LoRA
fine-tune, and converts the result to a fast, quantized format. It
produces a `live_model_ct2/` folder in your project directory — that's
the only output that matters; the script cleans up its own intermediate
files.

**This takes a while and needs several GB of free disk** (base model
download + a temporary merged copy). Run it once, locally or as a build
step in your deployment — not something end users ever trigger.

If it fails with a version-mismatch error between `ctranslate2` and
`transformers`, that's the most common failure mode — try:
```bash
pip install -U ctranslate2 transformers
```

## 3. Where the app looks for the converted model

By default: `live_model_ct2/` in the same folder as `streamlit_app.py`.
To point it somewhere else (e.g. a shared volume in production):

```bash
export LIVE_ASR_MODEL_PATH=/path/to/live_model_ct2
```

## 4. Turn it on

In the app, go to **Listen & Correct** → sidebar → toggle **🔴 Live
Correct (beta)**. If the model isn't converted yet, you'll see a clear
warning telling you to run step 2 — not a silent failure.

## 5. Browser requirements

- **Microphone access requires HTTPS** (or `localhost`) — Streamlit
  Community Cloud serves HTTPS by default, so this is fine there. If
  you're running locally via `streamlit run`, `http://localhost:8501` is
  also treated as a secure context by browsers, so mic access should
  still work.
- **The desktop app (pywebview shell)**: `http://127.0.0.1` is also
  treated as secure/localhost by the underlying browser engine, so this
  should work there too — but this specific combination (pywebview +
  WebRTC mic capture) is untested; see the checklist below.

## What I could not verify before handing this to you

I don't have a microphone, a browser, or real hardware in the
environment I built this in. I verified:
- Every file compiles cleanly
- The audio-buffer math (append/trim/threading) with synthetic numpy
  arrays standing in for real audio frames

I could NOT verify:
- That `streamlit-webrtc`'s actual frame format matches what
  `_resample_frame()` expects on your specific browser/OS
- Real end-to-end latency (my "1-3 second" estimate is based on
  faster-whisper's published benchmarks, not a measurement of this
  exact setup)
- That the model conversion script runs cleanly end-to-end (I couldn't
  download multi-GB model weights in this environment)

## Manual test checklist — please run through this yourself

- [ ] `python convert_model_for_live.py` completes and produces
      `live_model_ct2/`
- [ ] Toggling "Live Correct" on shows the webrtc widget and asks for
      mic permission
- [ ] Speaking into the mic shows *some* word highlighting appearing
      within a few seconds (doesn't need to be perfectly accurate yet —
      just confirm data is flowing end-to-end)
- [ ] "Stop & Submit" produces a final graded result and it shows up in
      session history same as a normal batch recording
- [ ] Test on the actual browser/OS combination your students will use
      — Safari's WebRTC behavior in particular sometimes differs from
      Chrome's

If step 2 or 3 breaks, send me the exact error and I'll fix the real
issue rather than guess at it.
