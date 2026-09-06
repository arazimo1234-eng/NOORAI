import glob
import json
import os


def load_quran(quranjson_root_dir):
    """
    Robust Quran loader that checks multiple common layouts including
    dist/chapters, dist/verses, or direct root json files.
    """
    quran = {}
    
    search_paths = [
        os.path.join(quranjson_root_dir, "dist", "verses", "*.json"),
        os.path.join(quranjson_root_dir, "dist", "chapters", "*.json"),
        os.path.join(quranjson_root_dir, "source", "surah", "surah_*.json"),
        os.path.join(quranjson_root_dir, "surahs", "*.json"),  # download_quran_data.py's layout
        os.path.join(quranjson_root_dir, "*.json")
    ]
    
    files = []
    for pattern in search_paths:
        files = glob.glob(pattern)
        if files:
            break
            
    if not files:
        raise FileNotFoundError(
            f"Could not find Quran text files under {quranjson_root_dir}. "
            "Make sure your JSON files are in the expected directory."
        )

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
                
            # Handle chapter/surah container files
            if isinstance(data, dict) and ("verses" in data or "ayahs" in data):
                surah_num = int(data.get("id", data.get("surah_number", data.get("index", 1))))
                name = data.get("transliteration") or data.get("name") or f"Surah {surah_num}"
                verses = {}
                verse_list = data.get("verses", data.get("ayahs", []))
                for v in verse_list:
                    if isinstance(v, dict):
                        v_id = int(v.get("id", v.get("verse_number", v.get("ayah", v.get("number", 1)))))
                        v_text = v.get("text", v.get("ayah_text", ""))
                        verses[v_id] = v_text
                quran[surah_num] = {"name": name, "verses": verses}
                
            # Handle single verse files (like dist/verses/*.json)
            elif isinstance(data, dict) and "text" in data:
                # Safely extract chapter/surah number handling possible nested dicts
                chap = data.get("chapter", data.get("surah_number", data.get("surah", 1)))
                if isinstance(chap, dict):
                    surah_num = int(chap.get("id", chap.get("number", 1)))
                else:
                    surah_num = int(chap)
                
                v_num_raw = data.get("verse_number", data.get("ayah", data.get("id", 1)))
                if isinstance(v_num_raw, dict):
                    verse_num = int(v_num_raw.get("id", v_num_raw.get("number", 1)))
                else:
                    verse_num = int(v_num_raw)
                    
                text = data.get("text", "")
                
                if surah_num not in quran:
                    quran[surah_num] = {"name": f"Surah {surah_num}", "verses": {}}
                quran[surah_num]["verses"][verse_num] = text

    if not quran:
        raise FileNotFoundError(f"Found files under {quranjson_root_dir} but failed to parse them.")
        
    return quran


def load_mutashabihat(path):
    if not path or not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    excluded = set()
    if isinstance(raw, dict):
        for surah_key, ayah_list in raw.items():
            try:
                surah_num = int(surah_key)
            except ValueError:
                continue
            if isinstance(ayah_list, list):
                for a in ayah_list:
                    try:
                        excluded.add((surah_num, int(a)))
                    except (ValueError, TypeError):
                        pass
    return excluded

