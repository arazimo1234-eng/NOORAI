"""
Quran Recitation & Evaluation App — Streamlit edition.

Architecture (fully local, no paid APIs):
  Browser mic  →  st.audio_input  →  local Whisper (Quranic fine-tune)
  →  Uthmani normalisation  →  dynamic window alignment
  →  word-level grading  →  Streamlit UI

Persistence (Stage 1): accounts, bookmarks, and every graded session are
stored in a local SQLite database (see db.py) — not a shared flat file and
not st.session_state alone, so data survives both a page refresh and more
than one concurrent user. See db.py's module docstring for schema details.

Model: MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix
  • Fine-tuned on tarteel-ai/everyayah + MohamedRashad/Quran-Recitations
  • Outputs Uthmani-diacritised Arabic (tashkeel preserved)
  • License: Apache 2.0 — commercial use permitted
  • Base: openai/whisper-large-v3-turbo

Fallback: tarteel-ai/whisper-base-ar-quran (lighter, ~150 MB)
  Switch by setting ASR_MODEL_ID below if RAM is tight.

Usage:
  pip install streamlit transformers torch numpy soundfile
  streamlit run streamlit_app.py
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import difflib
import io
import os
import random
import re
import unicodedata
from datetime import date

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import pipeline

# ── local ───────────────────────────────────────────────────────────────────
from quran_data import load_quran
import db
from progress_map import render_progress_map

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (edit these, nothing else needs changing)
# ════════════════════════════════════════════════════════════════════════════
QURANJSON_ROOT_DIR = "quran_repo/json/surahs"  # matches download_quran_data.py's OUT_DIR exactly

# Primary model: large-v3-turbo fine-tune — outputs with proper tashkeel.
# ~800 MB download on first run; needs ~2 GB RAM.
# Swap to the line below if your Chromebook runs out of memory:
#   ASR_MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
ASR_MODEL_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"

SAMPLE_RATE          = 16_000   # Whisper expects 16 kHz mono
CHUNK_LENGTH_S       = 30       # Whisper's native context window
STRIDE_LENGTH_S      = 5        # overlap between chunks (prevents boundary drops)
PASS_THRESHOLD       = 0.82     # overall similarity to count a segment as "correct"
CLOSE_WORD_THRESHOLD = 0.60     # per-word similarity: below → wrong, above → close
WINDOW_LOOKAHEAD     = 20       # ayahs ahead to include in the reference window

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VERSES_PER_PAGE = 20   # reader pagination, keeps long surahs (e.g. Al-Baqarah, 286 ayat) responsive

# ════════════════════════════════════════════════════════════════════════════
# BOOKMARKS  (Stage 1 fix: SQLite-backed, scoped to the logged-in user.
# The old version wrote every user's bookmarks into one shared bookmarks.json
# file, so two concurrent users silently overwrote each other's data. These
# thin wrappers keep the same call shape the rest of the file already uses,
# but every call is now scoped to st.session_state.user_id — see db.py for
# the actual persistence logic.)
# ════════════════════════════════════════════════════════════════════════════

def load_bookmarks() -> list:
    return db.load_bookmarks(st.session_state.user_id)


def add_bookmark(surah: int, ayah: int, surah_name: str, note: str = "") -> None:
    db.add_bookmark(st.session_state.user_id, surah, ayah, surah_name, note)


def remove_bookmark(surah: int, ayah: int) -> None:
    db.remove_bookmark(st.session_state.user_id, surah, ayah)


def is_bookmarked(surah: int, ayah: int, bookmarks: list) -> bool:
    return db.is_bookmarked(surah, ayah, bookmarks)


# ════════════════════════════════════════════════════════════════════════════
# ASR OUTPUT CLEANING
# Must run FIRST, before anything else touches the transcription string.
# ════════════════════════════════════════════════════════════════════════════

# Matches any Whisper-style control/metadata token: <|ar|>, <|transcribe|>,
# <|notimestamps|>, <|0.00|>, <|endoftext|>, etc. These are meant to be
# consumed by the tokenizer's skip_special_tokens logic, but LoRA fine-tunes
# frequently add new special tokens that the base processor doesn't know to
# suppress — so they leak into result["text"] as literal characters. Because
# they're emitted with no space before the first real word, they glue onto
# it (e.g. "<|ar|><|transcribe|><|notimestamps|>بِسْمِ") and that combined
# token fails to match anything in the reference window, which used to get
# scored as a wrong first word.
_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]*\|>")


def clean_asr_output(raw_text: str) -> str:
    """
    Strip Whisper control/metadata tags from raw ASR output.

    Removes every substring wrapped in `<|` and `|>` (language tag, task
    tag, timestamp tags, etc.), wherever they occur in the string — not
    just at the start, since some checkpoints emit timestamp tokens
    between segments too. Collapses any whitespace left behind by the
    removal so word-splitting downstream doesn't produce empty tokens.

    This is purely mechanical text cleanup — it does not touch Arabic
    characters, diacritics, or punctuation, and it is safe to call on
    already-clean text (idempotent).
    """
    if not raw_text:
        return ""
    cleaned = _CONTROL_TOKEN_RE.sub("", raw_text)
    # Defensive: also drop stray BOM / RTL-mark / LTR-mark characters that
    # some browser MediaRecorder → soundfile → Whisper paths introduce.
    cleaned = cleaned.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


# ════════════════════════════════════════════════════════════════════════════
# ARABIC NORMALISATION
# Two layers, kept strictly separate:
#   display_text  — the canonical Uthmani text shown to the user, NEVER mutated
#   norm(text)    — stripped skeleton used only for comparison
# ════════════════════════════════════════════════════════════════════════════

# All Unicode combining marks used in Uthmani Arabic (harakat, Quranic
# annotation symbols, tatweel, etc.)
_DIACRITICS = re.compile(
    r"[\u0610-\u061A"   # Arabic extended (Quranic annotation marks)
    r"\u064B-\u065F"   # Harakat (fathah … sukun)
    r"\u0670"          # Superscript alef (alef khanjariyya)
    r"\u06D6-\u06ED"   # Quranic annotation marks
    r"\u0640"          # Tatweel (kashida)
    r"]"
)

# Alef variants (with hamza above/below, with madda, with wasla) → plain alef
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

# Alif maqsura (looks like ya without dots) → ya, because some ASR models
# output ya where the reference has alif maqsura and vice-versa
_ALIF_MAQSURA = re.compile(r"ى")

# Teh marbuta → ha, to smooth over common transcription variants
_TEH_MARBUTA = re.compile(r"ة")


def normalize_arabic_text(text: str) -> str:
    """
    Produce a comparison skeleton from Arabic text.
    ONLY used for difflib matching — NEVER shown to the user and NEVER
    written back over the canonical Uthmani display_text.

    This is intentionally a one-way, lossy transform. Its job is to make
    two Uthmani-correct spellings that differ only in orthographic
    convention (not in which word they are) compare as equal, so that
    real mistakes aren't lost among false positives caused by encoding
    variance.

    Steps applied (in order):
      1. NFC normalise first (Unicode canonical composition) — some ASR
         checkpoints emit alef+hamza as two separate NFD codepoints
         (0627 + 0654) instead of the precomposed form (0623); collapsing
         to NFC up front means the diacritic/alef regexes below always see
         the composed forms they expect, rather than silently missing
         decomposed variants.
      2. Strip all diacritics and Quranic annotation marks (harakat,
         sukun in any of its encodings, Quranic pause/annotation symbols,
         tatweel/kashida elongation)
      3. Normalise alef variants (hamza-above/below, madda, wasla) → bare
         alef, since ASR models are inconsistent about which hamza seat
         they predict
      4. Normalise alif maqsura → ya (some checkpoints output ya where
         the Uthmani reference has alif maqsura, and vice versa)
      5. Normalise teh marbuta → ha (common ASR transcription variant)
      6. Collapse whitespace

    Returns the skeleton string. The original `text` argument is never
    mutated — normalize_arabic_text always returns a new string.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _DIACRITICS.sub("", t)
    t = _ALEF_VARIANTS.sub("ا", t)
    t = _ALIF_MAQSURA.sub("ي", t)
    t = _TEH_MARBUTA.sub("ه", t)
    t = " ".join(t.split())
    return t


# Backward-compatible alias — kept in case other modules in the project
# still import the old name.
normalise = normalize_arabic_text


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so it only happens once per Streamlit session)
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Quranic ASR model — first run downloads ~800 MB …")
def load_asr_pipeline():
    """
    Loads the Quranic Whisper model as a Hugging Face ASR pipeline.

    Using pipeline() (rather than raw WhisperForConditionalGeneration) because
    it handles chunked long-form audio natively via chunk_length_s /
    stride_length_s — preventing the silent 30-second truncation that happens
    when you feed raw audio directly to the feature extractor.
    """
    asr = pipeline(
        task="automatic-speech-recognition",
        model=ASR_MODEL_ID,
        device=DEVICE,
        # Tell the pipeline to emit word timestamps so we have alignment hooks
        # if we want to use them in future phases.
        return_timestamps=False,
        generate_kwargs={"language": "ar", "task": "transcribe"},
    )
    return asr


@st.cache_resource(show_spinner="Loading Quran text …")
def load_quran_data():
    # Self-healing: locally, you run download_quran_data.py once by hand
    # before starting the app. Streamlit Cloud has no such manual step —
    # it only ever runs streamlit_app.py — so the very first time this
    # boots on a fresh deploy (or after Cloud wipes the ephemeral
    # filesystem on a restart), the data simply isn't there yet. Rather
    # than requiring a step Cloud can't perform, fetch it here if missing.
    if not os.path.isdir(QURANJSON_ROOT_DIR) or not os.listdir(QURANJSON_ROOT_DIR):
        with st.spinner("First-time setup: downloading Quran text data …"):
            from download_quran_data import download_all
            download_all(out_dir=QURANJSON_ROOT_DIR)
    return load_quran(QURANJSON_ROOT_DIR)


# ════════════════════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_bytes: bytes, asr) -> str:
    """
    Decodes raw audio bytes (from st.audio_input), resamples if needed,
    and transcribes using the Quranic ASR pipeline with chunking.

    chunk_length_s=30 / stride_length_s=5:
      • Each chunk fits exactly inside Whisper's 30-second context window.
      • The 5-second overlap between chunks prevents words from being dropped
        at chunk boundaries (a common cause of false alignment failures).
    """
    audio_buf = io.BytesIO(audio_bytes)
    audio_array, original_sr = sf.read(audio_buf, dtype="float32")

    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample to 16 kHz if the browser delivered a different rate
    if original_sr != SAMPLE_RATE:
        try:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=original_sr, target_sr=SAMPLE_RATE
            )
        except ImportError:
            # Naive linear resample fallback if librosa isn't installed
            ratio = SAMPLE_RATE / original_sr
            new_len = int(len(audio_array) * ratio)
            indices = np.linspace(0, len(audio_array) - 1, new_len)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

    result = asr(
        {"array": audio_array, "sampling_rate": SAMPLE_RATE},
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        batch_size=1,
    )
    return result["text"].strip()


# ════════════════════════════════════════════════════════════════════════════
# ALIGNMENT & GRADING
# ════════════════════════════════════════════════════════════════════════════

def build_reference_window(sequence: list, position: int, max_ayahs: int) -> tuple:
    """
    Builds a flat word list from sequence[position : position + max_ayahs],
    together with a per-word owner index so we know which ayah each word came from.

    max_ayahs is caller-controlled (not always WINDOW_LOOKAHEAD) so the grader
    can start small and only widen the window when the recitation actually
    runs up against its edge — see grade_continuous_recitation for why this
    matters.

    Returns:
        window_words  — list of display-text words (Uthmani, never mutated)
        word_owner    — parallel list of (surah, ayah, seq_idx) per word
    """
    window_words, word_owner = [], []
    end = min(position + max_ayahs, len(sequence))
    for idx in range(position, end):
        surah_num, ayah_num, text = sequence[idx]
        for w in text.split():
            window_words.append(w)
            word_owner.append((surah_num, ayah_num, idx))
    return window_words, word_owner


def _align_window(window_words: list, transcribed_words: list) -> tuple:
    """
    Runs one difflib alignment pass of transcribed_words against window_words
    and returns (word_results, last_hit). Pulled out of grade_continuous_recitation
    so it can be re-run against progressively larger windows without duplicating
    the opcode-resolution logic.
    """
    norm_ref   = [normalize_arabic_text(w) for w in window_words]
    norm_trans = [normalize_arabic_text(w) for w in transcribed_words]

    matcher = difflib.SequenceMatcher(None, norm_ref, norm_trans, autojunk=False)
    word_results = [None] * len(window_words)
    last_hit = -1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                word_results[i] = "correct"
            if i2 > i1:
                last_hit = max(last_hit, i2 - 1)

        elif tag == "replace":
            ref_n, trans_n = i2 - i1, j2 - j1
            for offset in range(ref_n):
                i = i1 + offset
                if offset < trans_n:
                    j = j1 + offset
                    ratio = difflib.SequenceMatcher(
                        None, norm_ref[i], norm_trans[j], autojunk=False
                    ).ratio()
                    word_results[i] = "close" if ratio >= CLOSE_WORD_THRESHOLD else "wrong"
                else:
                    word_results[i] = "wrong"
            if ref_n > 0:
                last_hit = max(last_hit, i2 - 1)

        elif tag == "delete":
            for i in range(i1, i2):
                word_results[i] = "__delete__"

    return word_results, last_hit


def grade_continuous_recitation(transcribed: str, sequence: list, position: int) -> tuple:
    """
    Sliding-window sequence grader for continuous, multi-verse recitation.

    A user may keep reciting past the single ayah they were prompted with,
    so `transcribed` can legitimately span many verses in one go. This
    function does NOT try to grade one ayah at a time — it:

      1. Builds a flat, ordered word list ("reference window") starting at
         `position`, SIZED TO ROUGHLY HOW MUCH WAS ACTUALLY TRANSCRIBED —
         not a fixed 20-ayah block. This is deliberate: Quranic Arabic
         repeats short function words (و، من، لا، في …) constantly, and
         difflib.SequenceMatcher (with autojunk=False, which is needed so
         those common words aren't ignored entirely) will happily chain a
         match onto one of them several ayahs away by pure coincidence if
         given a large enough haystack. Handing it a 20-ayah window for a
         one-ayah recitation let exactly that happen: a stray match a few
         ayahs ahead pulled `last_hit` far past where the user actually
         stopped, and everything in between got graded "wrong"/"missing"
         instead of being left alone.
      2. Only WIDENS the window (and re-aligns) if the match runs right up
         against the current window's edge — that's the signal the user
         kept reciting past what we gave the matcher room for, not that a
         distant coincidental word match should be trusted. Widening stops
         at WINDOW_LOOKAHEAD ayahs.
      3. Finds the rightmost reference word actually reached
         (`last_hit`) and grades only up to that point — anything beyond
         is "not_recited" (unattempted), not "wrong". Session position
         only ever advances to just past `last_hit`.

    Key invariant:
        Words after the detected stop point are marked "not_recited", NOT
        wrong. The caller must NOT advance `position` if graded=False.

    Args:
        transcribed: raw (or already-cleaned) ASR output for this take.
                     clean_asr_output() is applied internally regardless,
                     so control tokens can never leak into alignment even
                     if a caller forgets to pre-clean.
        sequence:    ordered list of (surah_num, ayah_num, uthmani_text)
                     tuples — the full target range for the session.
        position:    index into `sequence` of the next ayah the user is
                     expected to recite.

    Returns: (result_dict, new_position)

    result_dict keys (when graded=True):
        graded          bool
        transcribed     str   — cleaned model output (control tokens stripped)
        start_surah     int
        start_ayah      int
        end_surah       int
        end_ayah        int
        ayahs           list  — [{surah, ayah, words: [{text, status}]}]
        similarity      float — 0–1, only over the graded portion
        passed          bool
    """
    transcribed = clean_asr_output(transcribed)

    if not transcribed:
        return {
            "graded": False,
            "reason": "Nothing was transcribed after removing model metadata tags.",
            "transcribed": transcribed,
        }, position

    transcribed_words = transcribed.split()
    max_ayah_count = min(WINDOW_LOOKAHEAD, len(sequence) - position)
    if max_ayah_count <= 0:
        return {"graded": False, "reason": "No reference text remaining."}, position

    # Start small: enough ayahs to plausibly hold ~2x the transcribed word
    # count (generous room for insertions/substitutions), at least 1 ayah.
    # Widen only if the alignment actually reaches this window's edge.
    ayah_count = 1
    window_words, word_owner, word_results, last_hit = [], [], [], -1

    while True:
        window_words, word_owner = build_reference_window(sequence, position, ayah_count)
        word_results, last_hit = _align_window(window_words, transcribed_words)

        reached_edge = last_hit >= len(window_words) - 1
        can_grow = ayah_count < max_ayah_count
        if reached_edge and can_grow:
            # Grow by enough to plausibly cover the rest of what was said,
            # not just +1 ayah at a time (avoids re-aligning ayah-by-ayah
            # for a long multi-ayah recitation).
            ayah_count = min(max_ayah_count, ayah_count * 2 + 2)
            continue
        break

    if last_hit == -1:
        # Nothing matched even in the fully-grown window — silence, noise,
        # or unrelated speech. Don't advance the session position.
        return {
            "graded": False,
            "reason": "Could not align with the expected passage. "
                      "Try reciting more clearly or adjusting the range.",
            "transcribed": transcribed,
        }, position

    # Resolve tentative "__delete__" marks
    for i in range(len(word_results)):
        if word_results[i] == "__delete__":
            # Before the stop-point: genuinely skipped mid-attempt.
            # At/after: simply not yet reached — leave as not_recited.
            word_results[i] = "missing" if i < last_hit else "not_recited"
        elif word_results[i] is None:
            word_results[i] = "not_recited"

    # Group words back by ayah
    ayah_map: dict = {}
    for i, (snum, anum, sidx) in enumerate(word_owner):
        ayah_map.setdefault(sidx, {"surah": snum, "ayah": anum, "words": []})
        ayah_map[sidx]["words"].append({"text": window_words[i], "status": word_results[i]})

    last_seq_idx  = word_owner[last_hit][2]
    new_position  = last_seq_idx + 1

    graded_ayahs = [
        ayah_map[idx]
        for idx in sorted(ayah_map)
        if idx <= last_seq_idx
    ]

    # Score only the graded portion (not not_recited words)
    graded_statuses = word_results[: last_hit + 1]
    matched = sum(1 for s in graded_statuses if s in ("correct", "close"))
    similarity = matched / len(graded_statuses) if graded_statuses else 0.0

    return {
        "graded":      True,
        "transcribed": transcribed,  # already cleaned of control tags
        "start_surah": sequence[position][0],
        "start_ayah":  sequence[position][1],
        "end_surah":   sequence[last_seq_idx][0],
        "end_ayah":    sequence[last_seq_idx][1],
        "ayahs":       graded_ayahs,
        "similarity":  similarity,
        "passed":      similarity >= PASS_THRESHOLD,
    }, new_position


# ════════════════════════════════════════════════════════════════════════════
# SEQUENCE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_sequence(quran: dict, from_surah: int, to_surah: int) -> list:
    """Returns an ordered list of (surah_num, ayah_num, text) tuples."""
    lo, hi = min(from_surah, to_surah), max(from_surah, to_surah)
    seq = []
    for s in sorted(quran):
        if lo <= s <= hi:
            for a in sorted(quran[s]["verses"]):
                seq.append((s, a, quran[s]["verses"][a]))
    return seq


# ════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════════════════

# Status → (CSS colour, label)
STATUS_STYLE = {
    "correct":     ("#7fa889", "✓"),
    "close":       ("#c9a769", "~"),
    "wrong":       ("#c17e6f", "✗"),
    "missing":     ("#93a0ac", "–"),
    "not_recited": ("#4a5560", " "),
}

def render_word(word: str, status: str) -> str:
    colour, _ = STATUS_STYLE.get(status, ("#e8e1d3", "?"))
    style = (
        f"color:{colour};"
        f"padding:0 2px;"
        f"border-radius:2px;"
        f"font-family:'Amiri',serif;"
        f"font-size:1.25rem;"
        f"line-height:2;"
    )
    if status == "wrong":
        style += "text-decoration:underline wavy #c17e6f;text-underline-offset:4px;"
    elif status == "close":
        style += "text-decoration:underline dotted #c9a769;text-underline-offset:4px;"
    elif status == "missing":
        style += "text-decoration:line-through;opacity:0.5;"
    elif status == "not_recited":
        style += "opacity:0.3;"
    return f'<span style="{style}" title="{status}">{word}</span>'


def render_ayah_block(ayah: dict) -> str:
    words_html = " ".join(
        render_word(w["text"], w["status"]) for w in ayah["words"]
    )
    header = (
        f'<div style="font-size:0.7rem;color:#93a0ac;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px;">'
        f'Surah {ayah["surah"]}, Ayah {ayah["ayah"]}</div>'
    )
    body = (
        f'<div style="direction:rtl;text-align:right;'
        f'padding:10px 12px;background:#1a2129;'
        f'border:1px solid #262f3a;border-radius:3px;">'
        f'{words_html}</div>'
    )
    return header + body


def legend_html() -> str:
    items = [
        ("correct",     "#7fa889", "Correct"),
        ("close",       "#c9a769", "Close (minor ASR variant)"),
        ("wrong",       "#c17e6f", "Wrong word"),
        ("missing",     "#93a0ac", "Skipped"),
        ("not_recited", "#4a5560", "Not yet reached"),
    ]
    chips = " &nbsp; ".join(
        f'<span style="color:{c};font-size:0.75rem;font-family:\'Inter\',sans-serif;">{label}</span>'
        for _, c, label in items
    )
    return f'<div style="margin-top:8px;">{chips}</div>'


# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Quran Recitation Tester",
    page_icon="📖",
    layout="centered",
)

db.init_db()

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Spectral:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown(
    """<style>
  :root {
    --ink-950: #0b0f14;
    --ink-900: #131920;
    --ink-800: #1a2129;
    --ink-700: #262f3a;
    --gilding: #c9a769;
    --gilding-dim: #6b5a3c;
    --parchment: #e8e1d3;
    --muted: #93a0ac;
    --sage: #7fa889;
    --terracotta: #c17e6f;
    --terracotta-dim: #8a4a3f;
  }

  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
  }

  /* ── Base ────────────────────────────────────────────────────────────── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--ink-950);
    color: var(--parchment);
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: var(--ink-900);
    border-right: 1px solid var(--ink-700);
  }
  [data-testid="stHeader"] { background: transparent; }

  h1, h2, h3 {
    font-family: 'Spectral', serif;
    font-weight: 600;
    color: var(--gilding);
    letter-spacing: -0.01em;
  }
  h4, h5, h6 { font-family: 'Spectral', serif; color: var(--parchment); }
  p, label, span, div { color: var(--parchment); }
  small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

  /* Visible keyboard focus everywhere — accessibility floor, not optional */
  *:focus-visible {
    outline: 2px solid var(--gilding) !important;
    outline-offset: 2px;
  }

  /* ── Signature element: illuminated divider ─────────────────────────────
     Every st.markdown("---") in the app renders as a plain <hr>. Restyled
     globally so the section-break grammar of the app matches the geometric
     medallions used to mark section breaks in illuminated Qur'an
     manuscripts — the same place a divider already existed, just no longer
     a generic flat line. */
  hr {
    border: none;
    height: 1px;
    margin: 1.75rem 0;
    background: linear-gradient(
      to right, transparent, var(--gilding-dim) 30%, var(--gilding-dim) 70%, transparent
    );
    position: relative;
  }
  hr::after {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    width: 7px; height: 7px;
    background: var(--gilding);
    transform: translate(-50%, -50%) rotate(45deg);
    box-shadow: 0 0 0 4px var(--ink-950);
  }

  /* ── Buttons ─────────────────────────────────────────────────────────── */
  .stButton > button {
    background: var(--gilding);
    color: var(--ink-950);
    border: none;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.01em;
    padding: 0.55rem 1.4rem;
    transition: background 0.15s ease, transform 0.1s ease;
  }
  .stButton > button:hover { background: #d8b87d; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }
  .stButton > button:disabled { background: var(--ink-700); color: var(--muted); }

  /* ── Text inputs, selects, date/text areas ──────────────────────────── */
  .stTextInput input, .stTextArea textarea, .stDateInput input,
  .stSelectbox > div > div, .stNumberInput input {
    background: var(--ink-800) !important;
    color: var(--parchment) !important;
    border: 1px solid var(--ink-700) !important;
    border-radius: 6px !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--gilding) !important;
    box-shadow: 0 0 0 1px var(--gilding) !important;
  }

  /* ── Mode selector: segmented-pill radio instead of default circles ──── */
  div[role="radiogroup"] {
    display: flex;
    gap: 4px;
    background: var(--ink-800);
    border: 1px solid var(--ink-700);
    border-radius: 8px;
    padding: 4px;
  }
  div[role="radiogroup"] label {
    flex: 1;
    text-align: center;
    border-radius: 5px;
    padding: 0.4rem 0.6rem;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s ease;
  }
  div[role="radiogroup"] label:has(input:checked) {
    background: var(--gilding);
    color: var(--ink-950) !important;
  }
  div[role="radiogroup"] label:has(input:checked) p { color: var(--ink-950) !important; }
  div[role="radiogroup"] input { position: absolute; opacity: 0; }

  /* ── Checkboxes / toggles ────────────────────────────────────────────── */
  .stCheckbox label span[data-baseweb="checkbox"],
  [data-testid="stToggle"] { accent-color: var(--gilding); }

  /* ── Tabs (login / create account) ──────────────────────────────────── */
  [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--ink-700); }
  [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--muted);
  }
  [data-baseweb="tab"][aria-selected="true"] {
    color: var(--gilding) !important;
    border-bottom-color: var(--gilding) !important;
  }

  /* ── Expanders — treat as quiet cards ───────────────────────────────── */
  [data-testid="stExpander"] {
    background: var(--ink-900);
    border: 1px solid var(--ink-700);
    border-radius: 8px;
  }
  [data-testid="stExpander"] summary { font-family: 'Inter', sans-serif; font-weight: 500; }

  /* ── Alerts (info/success/error) — quiet, tinted, not stock bootstrap ── */
  [data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"],
  [data-testid="stAlertContentError"] { font-family: 'Inter', sans-serif; font-size: 0.9rem; }

  .verdict-pass { color: var(--sage); font-family: 'Spectral', serif; font-weight: 600; font-size: 1.15rem; }
  .verdict-fail { color: var(--terracotta); font-family: 'Spectral', serif; font-weight: 600; font-size: 1.15rem; }

  .result-panel {
    border-left: 3px solid var(--gilding);
    padding: 12px 16px;
    margin: 10px 0;
    background: var(--ink-900);
    border-radius: 0 6px 6px 0;
  }

  /* ── Read Quran mode ─────────────────────────────────────────────────── */
  .verse-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 28px;
    padding: 0 6px;
    border: 1.5px solid var(--gilding);
    border-radius: 50%;
    color: var(--gilding);
    font-family: 'Spectral', serif;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    vertical-align: middle;
  }
  .verse-row {
    direction: rtl;
    text-align: right;
    font-family: 'Amiri', serif;
    font-size: clamp(1.4rem, 4vw, 1.85rem);
    line-height: 2.3;
    padding: 10px 4px;
    border-bottom: 1px solid var(--ink-700);
  }
  .surah-header {
    text-align: center;
    padding: 18px 0 22px;
    border-bottom: 2px solid var(--ink-700);
    margin-bottom: 12px;
  }
  .surah-header .arabic-name {
    font-family: 'Amiri', serif;
    font-size: clamp(1.7rem, 6vw, 2.2rem);
    color: var(--gilding);
  }
  .surah-header .meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .bookmark-chip {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--gilding);
    padding: 4px 0;
  }

  /* ── Brand lockup: logged-out screen ────────────────────────────────── */
  .brand-lockup { text-align: center; padding: 1.5rem 0 0.5rem; }
  .brand-mark {
    font-size: 2.4rem; color: var(--gilding); line-height: 1;
    margin-bottom: 0.25rem;
  }
  .brand-name {
    font-family: 'Spectral', serif; font-weight: 700; font-size: 2.4rem;
    color: var(--parchment); margin: 0; letter-spacing: -0.01em;
  }
  .brand-tagline {
    font-family: 'Amiri', serif; font-size: 1.1rem; color: var(--gilding);
    margin: 0.15rem 0 0.75rem;
  }
  .brand-sub {
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 1.25rem;
  }

  /* ── Sidebar brand lockup (compact, every screen once logged in) ──────── */
  .sidebar-brand {
    display: flex; align-items: center; gap: 8px;
    padding: 0.25rem 0 0.6rem;
  }
  .brand-mark-sm { font-size: 1.3rem; color: var(--gilding); }
  .brand-name-sm {
    font-family: 'Spectral', serif; font-weight: 700; font-size: 1.15rem;
    color: var(--parchment);
  }
  .sidebar-user { font-size: 0.82rem; color: var(--muted); margin-bottom: 0.6rem; }

  /* ── Main-area page header (replaces the old duplicated "Quran App" h2) ─ */
  .page-header {
    display: flex; align-items: baseline; gap: 10px;
    padding-bottom: 0.4rem; margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--ink-700);
  }
  .page-header-mark { font-size: 1.3rem; color: var(--gilding); }
  .page-header-name {
    font-family: 'Spectral', serif; font-weight: 700; font-size: 1.5rem;
    color: var(--parchment);
  }
  .page-header-tagline {
    font-family: 'Amiri', serif; font-size: 0.95rem; color: var(--muted);
  }

  /* ── Progress You Can See ──────────────────────────────────────────────
     Deliberately built as a plain CSS grid, not a chart library: this
     needs to render identically inside the Streamlit web engine AND
     inside the pywebview desktop shell with zero extra JS dependency. */
  .progress-summary-row {
    display: flex; gap: 0; margin: 0.5rem 0 1rem;
    border: 1px solid var(--ink-700); border-radius: 10px; overflow: hidden;
  }
  .progress-stat {
    flex: 1; text-align: center; padding: 0.9rem 0.5rem;
    border-right: 1px solid var(--ink-700); background: var(--ink-900);
  }
  .progress-stat:last-child { border-right: none; }
  .progress-stat-value {
    font-family: 'Spectral', serif; font-weight: 700; font-size: 1.6rem;
  }
  .progress-stat-unit { font-size: 0.9rem; color: var(--muted); font-weight: 400; }
  .progress-stat-label {
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-top: 2px;
  }
  .progress-due-banner {
    font-size: 0.85rem; background: var(--ink-900); border: 1px solid var(--gilding-dim);
    border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.75rem;
  }
  .progress-legend {
    display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.72rem;
    color: var(--muted); margin-bottom: 0.75rem;
  }
  .progress-legend span { display: inline-flex; align-items: center; gap: 5px; }
  .progress-legend i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
  .surah-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(46px, 1fr));
    gap: 5px;
    margin-bottom: 0.5rem;
  }
  .surah-tile {
    position: relative;
    aspect-ratio: 1;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    transition: transform 0.1s ease, box-shadow 0.1s ease;
    cursor: default;
  }
  .surah-tile:hover { transform: scale(1.12); box-shadow: 0 0 0 2px var(--parchment); z-index: 2; }
  .surah-tile-empty {
    background: transparent !important;
    border: 1px dashed var(--ink-700);
  }
  .tile-num {
    font-family: 'Inter', sans-serif; font-size: 0.62rem; font-weight: 600;
    color: var(--ink-950); opacity: 0.75;
  }
  .surah-tile-empty .tile-num { color: var(--muted); opacity: 0.6; }
  .tile-due-dot {
    position: absolute; top: -3px; right: -3px;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--terracotta); border: 1.5px solid var(--ink-950);
  }
  .weak-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0.7rem; background: var(--ink-900);
    border: 1px solid var(--ink-700); border-radius: 6px; font-size: 0.88rem;
  }
  .weak-row .muted { color: var(--muted); font-size: 0.78rem; }
  .weak-row-score { font-family: 'Spectral', serif; font-weight: 600; }

  /* ── Practice / Exam mode badge (Stage 3.2) ────────────────────────── */
  .mode-badge {
    display: inline-block; font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 0.72rem; letter-spacing: 0.03em; padding: 2px 9px;
    border-radius: 999px; vertical-align: middle;
  }
  .mode-badge-exam { background: rgba(201,167,105,0.18); color: var(--gilding); border: 1px solid var(--gilding-dim); }
  .mode-badge-practice { background: rgba(127,168,137,0.18); color: var(--sage); border: 1px solid var(--sage); }
</style>""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# Stage 1 fix: previously there were no accounts at all, so every bookmark
# and every recitation result belonged to nobody in particular — which is
# exactly why bookmarks.json collided across concurrent users and results
# had nowhere durable to live. Nothing below this block runs until
# st.session_state.user_id is set.
# ════════════════════════════════════════════════════════════════════════════

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None

if st.session_state.user_id is None:
    st.markdown(
        '<div class="brand-lockup">'
        '<div class="brand-mark">☾</div>'
        '<h1 class="brand-name">NoorHafiz</h1>'
        '<p class="brand-tagline">Light of the Memorizer</p>'
        '<p class="brand-sub">Smart review scheduling &middot; Listen &amp; Correct &middot; '
        "Progress you can see</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            uid = db.authenticate_user(login_username, login_password)
            if uid is not None:
                st.session_state.user_id = uid
                st.session_state.username = login_username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username", key="new_username")
            new_password = st.text_input("Choose a password", type="password", key="new_password")
            new_password_confirm = st.text_input(
                "Confirm password", type="password", key="new_password_confirm"
            )
            register_submitted = st.form_submit_button("Create account", use_container_width=True)
        if register_submitted:
            if not new_username.strip() or not new_password:
                st.error("Username and password cannot be empty.")
            elif new_password != new_password_confirm:
                st.error("Passwords don't match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                new_uid = db.create_user(new_username, new_password)
                if new_uid is None:
                    st.error("That username is already taken.")
                else:
                    st.session_state.user_id = new_uid
                    st.session_state.username = new_username.strip()
                    st.success("Account created — you're logged in.")
                    st.rerun()

    st.stop()  # nothing below this line executes for a logged-out visitor


# ════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP  (everything below only runs once st.session_state.user_id is set)
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark-sm">☾</span> '
        '<span class="brand-name-sm">NoorHafiz</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sidebar-user">Signed in as <strong>{st.session_state.username}</strong></div>',
        unsafe_allow_html=True,
    )
    if st.button("Log out", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

    # Stage 2.3: any account can opt into teacher mode — deliberately no
    # separate signup flow. Unlocks the "Teacher" mode below once enabled.
    is_teacher_now = db.is_teacher(st.session_state.user_id)
    new_teacher_value = st.checkbox("I'm a teacher", value=is_teacher_now)
    if new_teacher_value != is_teacher_now:
        db.set_teacher_status(st.session_state.user_id, new_teacher_value)
        st.rerun()

    st.markdown("---")

st.markdown(
    '<div class="page-header">'
    '<span class="page-header-mark">☾</span>'
    '<span class="page-header-name">NoorHafiz</span>'
    '<span class="page-header-tagline">Light of the Memorizer</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ── Load resources ───────────────────────────────────────────────────────────
# Quran text is cheap to load (local JSON) -- load eagerly, both modes need it.
# The ASR model is NOT loaded here -- it's ~800MB and only needed in Recite &
# Test mode, so it's loaded lazily inside render_recite_test_mode() instead.
quran = load_quran_data()
surah_numbers = sorted(quran.keys())
surah_options = {f"{n}. {quran[n]['name']}": n for n in surah_numbers}

# ── Session state defaults ───────────────────────────────────────────────────
defaults = {
    "sequence":       [],
    "position":       0,
    "results":        [],   # list of grade_continuous_recitation result dicts
    "session_live":   False,
    "app_mode":       "📖 Read Quran",
    "read_surah":     surah_numbers[0],
    "read_page":      0,
    "hifz_focus":     False,   # hides the starting hint in Recite & Test mode
    "test_mode":      "exam",  # "exam" (random start) or "practice" (sequential) — Stage 3.2
    "jump_from_surah": None,   # set by "Practice this surah" button in Read mode
    "jump_to_surah":   None,   # set alongside jump_from_surah when an assignment spans two different surahs
    "jump_start_ayah": None,   # set by the per-ayah "Recite from here" button in Read mode
    "active_assignment_id": None,  # set when a student starts a session via "Practice this assignment"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# READ QURAN MODE
# ════════════════════════════════════════════════════════════════════════════

def render_read_quran_mode(quran: dict, surah_options: dict, surah_numbers: list):
    bookmarks = load_bookmarks()

    with st.sidebar:
        st.markdown("### Navigate")
        read_label = st.selectbox(
            "Surah",
            list(surah_options.keys()),
            index=surah_numbers.index(st.session_state.read_surah),
            key="read_surah_select",
        )
        selected_surah = surah_options[read_label]
        if selected_surah != st.session_state.read_surah:
            st.session_state.read_surah = selected_surah
            st.session_state.read_page = 0  # reset pagination on surah change
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔖 Bookmarks")
        if not bookmarks:
            st.caption("No bookmarks yet. Tap 🔖 next to any ayah to save it.")
        else:
            for b in sorted(bookmarks, key=lambda x: (x["surah"], x["ayah"])):
                col1, col2 = st.columns([4, 1])
                with col1:
                    label = f"{b['surah_name']} {b['surah']}:{b['ayah']}"
                    if st.button(label, key=f"jump_{b['surah']}_{b['ayah']}", use_container_width=True):
                        st.session_state.read_surah = b["surah"]
                        st.session_state.read_page = (b["ayah"] - 1) // VERSES_PER_PAGE
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"rm_{b['surah']}_{b['ayah']}"):
                        remove_bookmark(b["surah"], b["ayah"])
                        st.rerun()

    surah_num = st.session_state.read_surah
    surah_name = quran[surah_num]["name"]
    verse_nums = sorted(quran[surah_num]["verses"].keys())
    total_verses = len(verse_nums)

    st.markdown(
        f'<div class="surah-header">'
        f'<div class="arabic-name">{surah_name}</div>'
        f'<div class="meta">Surah {surah_num} &middot; {total_verses} verses</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button(f"🎙 Practice Surah {surah_num} in Listen & Correct"):
        st.session_state.jump_from_surah = surah_num
        st.session_state.app_mode = "🎙 Listen & Correct"
        st.rerun()

    # ── Pagination (keeps long surahs like Al-Baqarah, 286 ayat, responsive) ──
    total_pages = max(1, (total_verses + VERSES_PER_PAGE - 1) // VERSES_PER_PAGE)
    page = min(st.session_state.read_page, total_pages - 1)
    start_idx = page * VERSES_PER_PAGE
    end_idx = min(start_idx + VERSES_PER_PAGE, total_verses)
    page_verse_nums = verse_nums[start_idx:end_idx]

    for ayah_num in page_verse_nums:
        text = quran[surah_num]["verses"][ayah_num]
        bookmarked = is_bookmarked(surah_num, ayah_num, bookmarks)

        col_text, col_bm, col_recite = st.columns([8, 1, 1])
        with col_text:
            st.markdown(
                f'<div class="verse-row">'
                f'<span class="verse-badge">{ayah_num}</span>{text}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_bm:
            icon = "🔖" if bookmarked else "🏷️"
            if st.button(icon, key=f"bm_{surah_num}_{ayah_num}", help="Toggle bookmark"):
                if bookmarked:
                    remove_bookmark(surah_num, ayah_num)
                else:
                    add_bookmark(surah_num, ayah_num, surah_name)
                st.rerun()
        with col_recite:
            if st.button("🎙", key=f"recite_{surah_num}_{ayah_num}", help="Recite from here"):
                # Jumps straight into Listen & Correct, pre-loads this surah
                # as the range, and forces the exact starting ayah — this is
                # deliberately NOT routed through Exam Mode's random start,
                # since clicking a specific ayah is an explicit request to
                # start exactly there, not to be tested from a random point.
                st.session_state.jump_from_surah = surah_num
                st.session_state.jump_to_surah = surah_num
                st.session_state.jump_start_ayah = ayah_num
                st.session_state.app_mode = "🎙 Listen & Correct"
                st.rerun()

    # ── Page navigation ──────────────────────────────────────────────────────
    if total_pages > 1:
        st.markdown("---")
        nav_prev, nav_label, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if page > 0 and st.button("← Previous", use_container_width=True):
                st.session_state.read_page = page - 1
                st.rerun()
        with nav_label:
            st.markdown(
                f'<p style="text-align:center;color:#93a0ac;font-size:0.8rem;">'
                f'Verses {start_idx + 1}–{end_idx} of {total_verses} '
                f'(page {page + 1} of {total_pages})</p>',
                unsafe_allow_html=True,
            )
        with nav_next:
            if page < total_pages - 1 and st.button("Next →", use_container_width=True):
                st.session_state.read_page = page + 1
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# RECITE & TEST MODE
# ════════════════════════════════════════════════════════════════════════════

def render_recite_test_mode(quran: dict, surah_options: dict, surah_numbers: list):
    # Lazy-load the ASR model only when this mode is actually used.
    asr = load_asr_pipeline()

    with st.sidebar:
        st.markdown("### Select Range")

        default_from_idx = 0
        default_to_idx = 0
        if st.session_state.jump_from_surah is not None:
            default_from_idx = surah_numbers.index(st.session_state.jump_from_surah)
            default_to_idx = default_from_idx
            if st.session_state.jump_to_surah is not None:
                default_to_idx = surah_numbers.index(st.session_state.jump_to_surah)
            st.session_state.jump_from_surah = None  # consume the one-time jump
            st.session_state.jump_to_surah = None

        from_label = st.selectbox(
            "From surah",
            list(surah_options.keys()),
            index=default_from_idx,
            disabled=st.session_state.session_live,
        )
        to_label = st.selectbox(
            "To surah",
            list(surah_options.keys()),
            index=default_to_idx,
            disabled=st.session_state.session_live,
        )

        from_surah = surah_options[from_label]
        to_surah   = surah_options[to_label]

        if st.session_state.jump_start_ayah is not None:
            st.caption(f"🎙 Will start exactly at Ayah {st.session_state.jump_start_ayah}.")

        st.markdown("---")
        st.markdown("### Mode")
        test_mode_label = st.radio(
            "Test mode",
            ["📝 Practice Mode", "🎯 Exam Mode"],
            index=1 if st.session_state.get("test_mode", "exam") == "exam" else 0,
            label_visibility="collapsed",
            disabled=st.session_state.session_live,
            help="Practice Mode starts at the beginning of your range. "
                 "Exam Mode drops you at a random ayah within the range — "
                 "the same way a real Hifz examiner tests, so you can't "
                 "just coast on the first few ayahs you always remember.",
        )
        st.session_state.test_mode = "exam" if "Exam" in test_mode_label else "practice"

        st.markdown("---")
        st.session_state.live_correct_on = st.toggle(
            "🔴 Live Correct (beta)",
            value=st.session_state.get("live_correct_on", False),
            disabled=st.session_state.session_live,
            help="Get corrections while you're still reciting instead of "
                 "waiting until you stop. Needs a one-time setup step on "
                 "the server — see LIVE_CORRECT_SETUP.md if this errors.",
        )

        st.markdown("---")
        st.session_state.hifz_focus = st.toggle(
            "🧠 Hifz Focus Mode",
            value=st.session_state.hifz_focus,
            help="Hides the starting-word hint, so you rely purely on memory rather than a visual cue.",
            disabled=st.session_state.session_live,
        )

        st.markdown("---")

        if not st.session_state.session_live:
            if st.button("▶  Start session", use_container_width=True):
                seq = build_sequence(quran, from_surah, to_surah)
                if seq:
                    if st.session_state.jump_start_ayah is not None:
                        # "Recite from here" from Read mode — an explicit
                        # starting point beats both Practice's sequential-
                        # from-0 and Exam's random start.
                        match = next(
                            (i for i, (s, a, _) in enumerate(seq)
                             if s == from_surah and a == st.session_state.jump_start_ayah),
                            0,
                        )
                        start_pos = match
                        st.session_state.test_mode = "practice"
                        st.session_state.jump_start_ayah = None  # consume the one-time jump
                    elif st.session_state.test_mode == "exam":
                        start_pos = random.randint(0, max(0, len(seq) - 1))
                    else:
                        start_pos = 0  # Practice Mode: sequential from the start of the range
                    st.session_state.sequence     = seq
                    st.session_state.position     = start_pos
                    st.session_state.results      = []
                    st.session_state.session_live = True
                    st.rerun()
                else:
                    st.error("No ayahs found in that range.")
        else:
            if st.button("⏹  End session", use_container_width=True):
                st.session_state.session_live = False
                st.rerun()

        st.markdown("---")
        st.markdown(
            f'<p style="font-size:0.7rem;color:#93a0ac;">'
            f'Model: <code style="color:#c9a769;">{ASR_MODEL_ID.split("/")[-1]}</code><br>'
            f'Device: <code style="color:#c9a769;">{DEVICE.upper()}</code><br>'
            f'Pass threshold: <code style="color:#c9a769;">{int(PASS_THRESHOLD * 100)}%</code>'
            f"</p>",
            unsafe_allow_html=True,
        )

    if not st.session_state.session_live:
        if st.session_state.active_assignment_id is not None:
            st.markdown("📋 **Completing a teacher assignment** — your result will be submitted for review.")
        st.info(
            "Select a surah range in the sidebar and press **▶ Start session** "
            "to begin. You will be given a starting ayah — recite from memory "
            "for as long as you like, then submit."
        )
        return

    seq      = st.session_state.sequence
    position = st.session_state.position

    if position >= len(seq):
        st.success("🎉 You have reached the end of the selected range!")
        st.session_state.session_live = False
        return

    cur_surah, cur_ayah, cur_text = seq[position]
    total = len(seq)
    done  = position

    st.markdown(
        f'<p style="font-size:0.75rem;color:#93a0ac;">'
        f'Ayah {done + 1} of {total} &nbsp;·&nbsp; '
        f'Surah {cur_surah} ({quran[cur_surah]["name"]}), Ayah {cur_ayah}'
        f"</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.hifz_focus:
        st.markdown(
            '<div class="result-panel" style="font-family:Amiri,serif;'
            'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;'
            'color:#4a5568;">'
            "🧠 Hifz Focus Mode — recite from memory, no hint shown"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        hint_words = " ".join(cur_text.split()[:3])
        st.markdown(
            f'<div class="result-panel" style="font-family:Amiri,serif;'
            f'direction:rtl;text-align:right;font-size:1.6rem;line-height:1.8;">'
            f"{hint_words} …"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.caption("Complete this ayah (and continue reciting as many as you like).")

    if st.session_state.get("live_correct_on", False):
        render_live_recording(quran, asr, seq, position)
    else:
        render_batch_recording(quran, asr, seq, position)


def finalize_recitation_result(quran: dict, result: dict, seq: list, new_position: int) -> None:
    """
    Shared by BOTH recording paths — the original record-then-submit flow
    and the new Live Correct flow's final "Stop & Submit" pass. Persists
    the graded result and renders the verdict/word breakdown exactly the
    same way regardless of which path produced it, so a teacher reviewing
    submissions later can't tell (and shouldn't be able to tell) which
    recording mode a student used.
    """
    st.session_state.results.append(result)
    st.session_state.position = new_position
    session_id = db.save_session_result(
        st.session_state.user_id, result, mode=st.session_state.get("test_mode", "exam")
    )
    db.seed_review_from_session(st.session_state.user_id, result)
    if st.session_state.active_assignment_id is not None:
        db.record_submission(st.session_state.active_assignment_id, st.session_state.user_id, session_id)
        st.session_state.active_assignment_id = None
        st.success("Submitted to your teacher for review.")

    pct = result["similarity"] * 100
    passed = result["passed"]

    st.markdown("---")
    mode_badge = (
        '<span class="mode-badge mode-badge-exam">🎯 Exam Mode</span>'
        if st.session_state.get("test_mode", "exam") == "exam"
        else '<span class="mode-badge mode-badge-practice">📝 Practice Mode</span>'
    )
    st.markdown(f'#### Result &nbsp;{mode_badge}', unsafe_allow_html=True)

    if passed:
        st.markdown(
            f'<p class="verdict-pass">✓ Correct &nbsp;·&nbsp; {pct:.1f}% similarity</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p class="verdict-fail">✗ Needs review &nbsp;·&nbsp; {pct:.1f}% similarity</p>',
            unsafe_allow_html=True,
        )

    if result["start_surah"] == result["end_surah"] and result["start_ayah"] == result["end_ayah"]:
        range_str = f"Surah {result['start_surah']}, Ayah {result['start_ayah']}"
    else:
        range_str = (
            f"Surah {result['start_surah']}:{result['start_ayah']} "
            f"→ {result['end_surah']}:{result['end_ayah']}"
        )
    st.caption(f"Graded: {range_str}")

    st.markdown(legend_html(), unsafe_allow_html=True)
    for ayah in result["ayahs"]:
        st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)

    with st.expander("Raw ASR transcription"):
        st.write(result["transcribed"])

    st.markdown("---")

    if new_position >= len(seq):
        st.success("🎉 You have completed the entire selected range!")
        st.session_state.session_live = False
    else:
        next_s, next_a, _ = seq[new_position]
        remaining = len(seq) - new_position
        st.info(
            f"Next: Surah {next_s} ({quran[next_s]['name']}), "
            f"Ayah {next_a} &nbsp;·&nbsp; {remaining} ayah(s) remaining."
        )
        if st.button("Continue to next ayah →", key="continue_btn", type="primary", use_container_width=True):
            st.rerun()


def render_batch_recording(quran: dict, asr, seq: list, position: int) -> None:
    """The original record-a-clip-then-submit flow, unchanged in behavior —
    extracted verbatim into its own function so Live Correct can sit next
    to it as an alternative without duplicating either code path."""
    st.markdown("#### 🎙 Record your recitation")
    audio_value = st.audio_input(
        label="Press the microphone button, recite, then press Stop",
        key=f"audio_{position}",
    )

    if audio_value is None:
        return

    audio_bytes = audio_value.read()

    with st.spinner("Transcribing with the Quranic ASR model …"):
        try:
            transcription = transcribe_audio(audio_bytes, asr)
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            return

    transcription = clean_asr_output(transcription)

    if not transcription.strip():
        st.warning(
            "Nothing was transcribed — the recording may have been too quiet "
            "or too short. Try again."
        )
        return

    result, new_position = grade_continuous_recitation(transcription, seq, position)

    if not result["graded"]:
        st.warning(result.get("reason", "Could not align. Try again."))
        with st.expander("Raw transcription"):
            st.write(transcription)
        return

    finalize_recitation_result(quran, result, seq, new_position)


def render_live_recording(quran: dict, asr, seq: list, position: int) -> None:
    """
    Live Correct (beta) — continuous mic capture via streamlit-webrtc,
    fast approximate re-grading every ~2 seconds via faster-whisper, and
    a final accurate pass through the SAME batch model/grader used
    everywhere else once the user stops. See live_asr.py for the full
    design rationale and the honest limitations (latency, one-time model
    conversion requirement, untested-against-real-hardware caveat).
    """
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode
        from streamlit_autorefresh import st_autorefresh
        import live_asr
    except ImportError:
        st.error(
            "Live Correct needs extra packages that aren't installed. "
            "Run `pip install -r requirements_live.txt`, then restart the app."
        )
        return

    buffer_key = f"live_buffer_{position}"
    if buffer_key not in st.session_state:
        st.session_state[buffer_key] = live_asr.LiveAudioBuffer()
    buffer: live_asr.LiveAudioBuffer = st.session_state[buffer_key]

    st.markdown("#### 🔴 Live Correct — recite naturally, corrections update as you go")
    st.caption(
        "Live feedback below uses a faster, approximate pass for speed. "
        "Press **Stop & Submit** when done for the accurate, saved grade."
    )

    webrtc_ctx = webrtc_streamer(
        key=f"live_webrtc_{position}",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=1024,
    )

    if webrtc_ctx.state.playing:
        st_autorefresh(interval=2000, key=f"live_refresh_{position}")
        live_asr.poll_webrtc_frames(webrtc_ctx, buffer, timeout=1.0)

        if buffer.new_seconds_available() >= live_asr.TICK_MIN_NEW_SECONDS:
            try:
                model = live_asr.load_live_asr_model()
            except RuntimeError as exc:
                st.warning(str(exc))
                model = None

            if model is not None:
                snapshot = buffer.snapshot()
                buffer.mark_read()
                with st.spinner("Updating live transcript …"):
                    live_text = live_asr.transcribe_buffer(model, snapshot)
                if live_text:
                    live_result, _ = grade_continuous_recitation(live_text, seq, position)
                    st.session_state[f"live_preview_{position}"] = live_result

        preview = st.session_state.get(f"live_preview_{position}")
        if preview and preview.get("graded"):
            st.markdown(legend_html(), unsafe_allow_html=True)
            for ayah in preview["ayahs"]:
                st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)
        else:
            st.info("Listening — start reciting, corrections will appear here.")

    total_seconds = buffer.total_seconds()
    if total_seconds > 0:
        st.caption(f"{total_seconds:.0f}s captured")

    if st.button("⏹ Stop & Submit", type="primary", use_container_width=True, disabled=total_seconds == 0):
        final_audio = buffer.snapshot()
        buffer.clear()
        with st.spinner("Running the final, accurate grading pass …"):
            # Deliberately the SAME accurate batch model used everywhere
            # else in the app for the saved/scored result — Live Correct's
            # fast model is for in-the-moment feedback only, never for
            # what gets written to the database or a teacher's queue.
            audio_buf_wav = _float_array_to_wav_bytes(final_audio)
            transcription = clean_asr_output(transcribe_audio(audio_buf_wav, asr))

        if not transcription.strip():
            st.warning("Nothing was transcribed — try again.")
            return

        result, new_position = grade_continuous_recitation(transcription, seq, position)
        if not result["graded"]:
            st.warning(result.get("reason", "Could not align. Try again."))
            return

        finalize_recitation_result(quran, result, seq, new_position)


def _float_array_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Converts the live buffer's float32 @ 16kHz array into WAV bytes so
    it can go through the exact same transcribe_audio() the batch path
    uses, rather than maintaining a second array-shaped code path."""
    buf = io.BytesIO()
    sf.write(buf, audio, live_asr_sample_rate(), format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def live_asr_sample_rate() -> int:
    from live_asr import SAMPLE_RATE
    return SAMPLE_RATE




    # ── All-time practice history ────────────────────────────────────────────
    # Stage 1 fix: this reads from the database, not st.session_state, so it
    # is still here after a refresh, a closed tab, or a new login tomorrow —
    # unlike the "this session" expander just below it, which (by design)
    # only covers segments graded since the current session started.
    st.markdown("---")

    # Stage 2.2 — streaks + trend. Shown outside the expander, right after
    # a graded result, since this is exactly the moment "come back
    # tomorrow" needs reinforcing — burying it in a collapsed section
    # would defeat the point of building it at all.
    streaks = db.get_streaks(st.session_state.user_id)
    if streaks["current_streak"] > 0 or streaks["longest_streak"] > 0:
        streak_col1, streak_col2 = st.columns(2)
        with streak_col1:
            flame = "🔥" if streaks["active_today"] else "💤"
            st.metric(f"{flame} Current streak", f"{streaks['current_streak']} day(s)")
        with streak_col2:
            st.metric("🏆 Longest streak", f"{streaks['longest_streak']} day(s)")
        if not streaks["active_today"] and streaks["current_streak"] > 0:
            st.caption("Recite today to keep your streak alive.")

        trend = db.get_similarity_trend(st.session_state.user_id, limit_days=30)
        if len(trend) >= 2:
            st.caption("Accuracy trend — daily average similarity, last 30 active days")
            chart_data = {r["date"]: r["similarity"] * 100 for r in trend}
            st.line_chart(chart_data, height=180)

    overall = db.get_overall_stats(st.session_state.user_id)
    if overall["n_sessions"] > 0:
        with st.expander(f"📊 All-time practice history ({overall['n_sessions']} segment(s) ever graded)"):
            col1, col2 = st.columns(2)
            col1.metric("Overall similarity (all-time)", f"{overall['avg_similarity'] * 100:.1f}%")
            col2.metric("Pass rate (all-time)", f"{overall['pass_rate'] * 100:.1f}%")

            weak = db.get_weak_ayahs(st.session_state.user_id, limit=10)
            if weak:
                st.caption("Most-missed ayahs across all your sessions:")
                for w in weak:
                    name = quran.get(w["surah"], {}).get("name", f"Surah {w['surah']}")
                    st.markdown(
                        f"- **{name} {w['surah']}:{w['ayah']}** — missed in "
                        f"{w['miss_count']} word instance(s)"
                    )

            recent = db.get_session_history(st.session_state.user_id, limit=20)
            st.caption("Recent sessions:")
            for r in recent:
                label = "✓" if r["passed"] else "✗"
                mode_tag = "🎯" if r.get("mode", "exam") == "exam" else "📝"
                st.markdown(
                    f"{label} {mode_tag} {r['start_surah']}:{r['start_ayah']} → "
                    f"{r['end_surah']}:{r['end_ayah']} — "
                    f"{r['similarity'] * 100:.1f}% — {r['created_at'][:10]}"
                )

    if st.session_state.results:
        with st.expander(f"Session history ({len(st.session_state.results)} segment(s))"):
            total_scored = sum(
                len([w for a in r["ayahs"] for w in a["words"]
                     if w["status"] != "not_recited"])
                for r in st.session_state.results
            )
            total_correct = sum(
                len([w for a in r["ayahs"] for w in a["words"]
                     if w["status"] in ("correct", "close")])
                for r in st.session_state.results
            )
            overall_pct = (total_correct / total_scored * 100) if total_scored else 0
            st.metric("Overall similarity (this session)", f"{overall_pct:.1f}%")

            for i, r in enumerate(reversed(st.session_state.results), 1):
                pct = r["similarity"] * 100
                label = "✓" if r["passed"] else "✗"
                with st.expander(
                    f"{label} Segment {len(st.session_state.results) - i + 1} — "
                    f"{r['start_surah']}:{r['start_ayah']} → "
                    f"{r['end_surah']}:{r['end_ayah']} — {pct:.1f}%"
                ):
                    for ayah in r["ayahs"]:
                        st.markdown(render_ayah_block(ayah), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TODAY'S REVIEW  (Stage 2.1 — the SRS queue)
# Deliberately rendered before the mode radio/dispatch below, per the
# roadmap's own instruction: this should be the first thing a logged-in
# user sees, not buried in a tab.
# ════════════════════════════════════════════════════════════════════════════

def render_todays_review(quran: dict) -> None:
    due = db.get_due_reviews(st.session_state.user_id, limit=20)
    if not due:
        return  # nothing due — don't clutter the page with an empty state

    with st.container():
        st.markdown(f"### 🔁 Today's Review — {len(due)} ayah(s) due")
        st.caption(
            "These came up wrong, close, or unattempted before. Recite them again "
            "in Listen & Correct mode to clear them from this list."
        )
        for item in due:
            name = quran.get(item["surah"], {}).get("name", f"Surah {item['surah']}")
            overdue_days = (date.today() - item["next_due_date"]).days if item["next_due_date"] else 0
            overdue_note = f" — {overdue_days} day(s) overdue" if overdue_days > 0 else ""
            st.markdown(f"- **{name} {item['surah']}:{item['ayah']}**{overdue_note}")
        st.markdown("---")


render_todays_review(quran)


# ════════════════════════════════════════════════════════════════════════════
# MY ASSIGNMENTS  (Stage 2.3, student-facing side)
# ════════════════════════════════════════════════════════════════════════════

def render_my_assignments(quran: dict) -> None:
    student_id = st.session_state.user_id

    with st.expander("🏫 Join a classroom"):
        code = st.text_input("Enter join code from your teacher", key="join_code_input")
        if st.button("Join") and code.strip():
            success, message = db.join_classroom(student_id, code)
            (st.success if success else st.error)(message)
            if success:
                st.rerun()

    assignments = db.get_student_assignments(student_id)
    pending = [a for a in assignments if not a["already_submitted"]]
    if not pending:
        return

    st.markdown(f"### 📋 Assignments — {len(pending)} pending")
    for a in pending:
        from_name = quran.get(a["from_surah"], {}).get("name", str(a["from_surah"]))
        to_name = quran.get(a["to_surah"], {}).get("name", str(a["to_surah"]))
        due_str = f" — due {a['due_date']}" if a["due_date"] else ""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{a['classroom_name']}**: {from_name} → {to_name}{due_str}")
        with col2:
            if st.button("Start", key=f"start_assignment_{a['id']}"):
                st.session_state.jump_from_surah = a["from_surah"]
                st.session_state.jump_to_surah = a["to_surah"]
                st.session_state.active_assignment_id = a["id"]
                st.session_state.app_mode = "🎙 Listen & Correct"
                st.rerun()
    st.markdown("---")


render_my_assignments(quran)



# ════════════════════════════════════════════════════════════════════════════
# PROGRESS MODE  (Stage 2.2 — "Progress You Can See" from the pitch)
# ════════════════════════════════════════════════════════════════════════════

def render_progress_mode(quran: dict, surah_numbers: list) -> None:
    st.markdown("### Your memorization map")
    st.caption(
        "Color-coded by strength, built entirely from your real Listen & "
        "Correct history — not a projection."
    )
    render_progress_map(quran, surah_numbers)


# ════════════════════════════════════════════════════════════════════════════
# TEACHER MODE  (Stage 2.3)
# ════════════════════════════════════════════════════════════════════════════

def render_teacher_mode(quran: dict, surah_options: dict, surah_numbers: list):
    teacher_id = st.session_state.user_id

    st.markdown("#### Your classrooms")
    classrooms = db.get_teacher_classrooms(teacher_id)

    with st.expander("➕ Create a new classroom"):
        new_name = st.text_input("Classroom name", key="new_classroom_name")
        if st.button("Create classroom") and new_name.strip():
            created = db.create_classroom(teacher_id, new_name)
            st.success(f"Created **{new_name}** — join code: `{created['join_code']}`")
            st.caption("Share this code with your students so they can join.")
            st.rerun()

    if not classrooms:
        st.info("You haven't created a classroom yet — use the panel above to start one.")
        return

    classroom_labels = {f"{c['name']} ({c['member_count']} student(s))": c for c in classrooms}
    selected_label = st.selectbox("Select a classroom", list(classroom_labels.keys()))
    classroom = classroom_labels[selected_label]

    st.markdown(
        f'<p style="font-size:0.8rem;color:#93a0ac;">Join code: '
        f'<code style="color:#c9a769;">{classroom["join_code"]}</code></p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Assignments")

    with st.expander("➕ New assignment"):
        col1, col2 = st.columns(2)
        with col1:
            from_label = st.selectbox("From surah", list(surah_options.keys()), key="assign_from")
        with col2:
            to_label = st.selectbox("To surah", list(surah_options.keys()), key="assign_to")
        due = st.date_input("Due date (optional)", value=None, key="assign_due")
        if st.button("Create assignment"):
            db.create_assignment(
                classroom["id"],
                surah_options[from_label],
                surah_options[to_label],
                due if due else None,
            )
            st.success("Assignment created.")
            st.rerun()

    assignments = db.get_classroom_assignments(classroom["id"])
    if not assignments:
        st.info("No assignments yet for this classroom.")
        return

    assignment_labels = {}
    for a in assignments:
        from_name = quran.get(a["from_surah"], {}).get("name", str(a["from_surah"]))
        to_name = quran.get(a["to_surah"], {}).get("name", str(a["to_surah"]))
        due_str = f" — due {a['due_date']}" if a["due_date"] else ""
        assignment_labels[f"{from_name} → {to_name}{due_str}"] = a

    selected_assignment_label = st.selectbox("View submissions for", list(assignment_labels.keys()))
    assignment = assignment_labels[selected_assignment_label]

    submissions = db.get_assignment_submissions(assignment["id"])
    if not submissions:
        st.info("No submissions yet for this assignment.")
        return

    st.markdown(f"**{len(submissions)} submission(s)**")
    for sub in submissions:
        status_icon = "✓" if sub["passed"] else "✗"
        reviewed_note = " · reviewed" if sub["teacher_reviewed"] else " · **needs review**"
        with st.expander(
            f"{status_icon} {sub['username']} — {sub['similarity'] * 100:.1f}%{reviewed_note}"
        ):
            words = db.get_session_words(sub["session_id"])
            # Reuse the exact same rendering the student sees — the teacher
            # is looking at the same graded output, not a different summary.
            by_ayah: dict = {}
            for w in words:
                key = (w["surah"], w["ayah"])
                by_ayah.setdefault(key, {"surah": w["surah"], "ayah": w["ayah"], "words": []})
                by_ayah[key]["words"].append({"text": w["word_text"], "status": w["status"]})
            for ayah_dict in by_ayah.values():
                st.markdown(render_ayah_block(ayah_dict), unsafe_allow_html=True)

            comment = st.text_area(
                "Comment for the student",
                value=sub["teacher_comment"] or "",
                key=f"comment_{sub['submission_id']}",
            )
            if st.button("Save comment", key=f"save_{sub['submission_id']}"):
                db.update_submission_review(sub["submission_id"], comment)
                st.success("Saved.")
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MODE DISPATCH
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Mode")
    mode_options = ["📖 Read Quran", "🎙 Listen & Correct", "📊 Progress"]
    if is_teacher_now:
        mode_options.append("🧑‍🏫 Teacher")
    if st.session_state.app_mode in ("🎙 Recite & Test",):  # legacy label migration
        st.session_state.app_mode = "🎙 Listen & Correct"
    if st.session_state.app_mode not in mode_options:
        st.session_state.app_mode = mode_options[0]
    mode = st.radio(
        "Choose a mode",
        mode_options,
        index=mode_options.index(st.session_state.app_mode),
        label_visibility="collapsed",
    )
    if mode != st.session_state.app_mode:
        st.session_state.app_mode = mode
        st.rerun()
    st.markdown("---")

if st.session_state.app_mode == "📖 Read Quran":
    render_read_quran_mode(quran, surah_options, surah_numbers)
elif st.session_state.app_mode == "🎙 Listen & Correct":
    render_recite_test_mode(quran, surah_options, surah_numbers)
elif st.session_state.app_mode == "📊 Progress":
    render_progress_mode(quran, surah_numbers)
else:
    render_teacher_mode(quran, surah_options, surah_numbers)
