"""
One-time setup: convert the recitation model into CTranslate2 format so
Live Correct can run fast enough on CPU to feel real-time.

WHY THIS IS A SEPARATE STEP, NOT AUTOMATIC ON APP STARTUP
------------------------------------------------------------
This does real work (loading the full model, merging LoRA weights,
re-exporting, quantizing) that takes real time and disk space, and it
only needs to happen ONCE per model version, not on every app boot. Doing
it lazily inside the Streamlit app would mean the first person to open
Live Correct after a deploy sits through a multi-minute conversion with
no clear explanation — running it here, explicitly, with progress output,
is the honest version of that same work.

WHAT THIS ACTUALLY DOES
-------------------------
`ASR_MODEL_ID` (MaddoggProduction/whisper-l-v3-turbo-quran-lora-...) is a
LoRA ADAPTER on top of a base Whisper checkpoint, not a standalone model.
CTranslate2's converter needs a standalone (merged) Hugging Face model
directory as input, so this script:

  1. Downloads the base Whisper model + the LoRA adapter
  2. Merges the adapter into the base weights (peft's merge_and_unload)
  3. Saves the merged model to a local folder
  4. Runs ct2-transformers-converter against that folder, quantized to
     int8 (the fast-and-small option that's the whole point of this file)

Run once:
    python convert_model_for_live.py

Output:
    ./live_model_ct2/   <- point LIVE_ASR_MODEL_PATH at this (it's the
                            default, so no env var needed if you run this
                            from the project root)

REQUIRES (add to your venv once, not part of the app's normal
requirements — this script itself is only ever run by a developer, never
by an end user or the deployed app):
    pip install peft ctranslate2 transformers[torch]
"""

import os
import shutil
import subprocess
import sys

BASE_MODEL_CANDIDATES = [
    # The LoRA adapter's config names its base model — this list is a
    # fallback in case that lookup fails; the script tries the adapter's
    # own declared base first, always.
    "openai/whisper-large-v3-turbo",
]
LORA_ADAPTER_ID = "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix"
MERGED_DIR = "merged_model_tmp"
OUTPUT_DIR = "live_model_ct2"


def _get_declared_base_model(adapter_id: str) -> str | None:
    try:
        from peft import PeftConfig
        cfg = PeftConfig.from_pretrained(adapter_id)
        return cfg.base_model_name_or_path
    except Exception as exc:
        print(f"Could not read the adapter's declared base model ({exc}); "
              f"falling back to the candidate list.")
        return None


def merge_lora_into_base() -> str:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    base_id = _get_declared_base_model(LORA_ADAPTER_ID) or BASE_MODEL_CANDIDATES[0]
    print(f"Base model: {base_id}")
    print(f"LoRA adapter: {LORA_ADAPTER_ID}")

    print("Downloading + loading base model (this is the large download)…")
    base_model = WhisperForConditionalGeneration.from_pretrained(base_id)
    processor = WhisperProcessor.from_pretrained(base_id)

    print("Applying LoRA adapter…")
    merged = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
    merged = merged.merge_and_unload()

    if os.path.isdir(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR, exist_ok=True)
    print(f"Saving merged standalone model to ./{MERGED_DIR}/ …")
    merged.save_pretrained(MERGED_DIR)
    processor.save_pretrained(MERGED_DIR)
    return MERGED_DIR


def convert_to_ctranslate2(merged_dir: str) -> None:
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    print(f"Converting to CTranslate2 (int8) at ./{OUTPUT_DIR}/ …")
    result = subprocess.run(
        [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", merged_dir,
            "--output_dir", OUTPUT_DIR,
            "--quantization", "int8",
        ],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            "ct2 conversion failed — see the error above. Common cause: "
            "the `ctranslate2` package version doesn't match the "
            "`transformers` version installed. Try "
            "`pip install -U ctranslate2 transformers`."
        )
    print("Done. Live Correct will now find the model at "
          f"./{OUTPUT_DIR}/ automatically.")


def main() -> None:
    merged_dir = merge_lora_into_base()
    convert_to_ctranslate2(merged_dir)
    print("\nCleaning up the intermediate merged model (kept the small "
          "CTranslate2 output, deleted the large intermediate copy)…")
    shutil.rmtree(merged_dir, ignore_errors=True)
    print("Setup complete.")


if __name__ == "__main__":
    main()
