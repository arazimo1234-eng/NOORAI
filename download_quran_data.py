"""
Downloads the 114 surah JSON files from the risan/quran-json project
(via the jsDelivr CDN build) into quran_repo/json/surahs/, matching the
folder that quran_data.load_quran() expects.

Run this once per Colab session (the filesystem resets every runtime),
BEFORE running streamlit_app.py.
"""
import os
import time
import urllib.request
import urllib.error

BASE_URL = "https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/chapters"
OUT_DIR = "quran_repo/json/surahs"


def download_all(out_dir=OUT_DIR, retries=3):
    os.makedirs(out_dir, exist_ok=True)
    failed = []

    for n in range(1, 115):
        out_path = os.path.join(out_dir, f"{n}.json")
        if os.path.exists(out_path):
            continue  # already downloaded this session

        url = f"{BASE_URL}/{n}.json"
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=15) as resp:
                    data = resp.read()
                with open(out_path, "wb") as f:
                    f.write(data)
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == retries:
                    failed.append((n, str(e)))
                else:
                    time.sleep(1)

    total = len(os.listdir(out_dir))
    print(f"Downloaded {total}/114 surah files into '{out_dir}'.")
    if failed:
        print(f"Failed to download {len(failed)} file(s): {failed}")


if __name__ == "__main__":
    download_all()
