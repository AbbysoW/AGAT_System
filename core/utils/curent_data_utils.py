import json
from pathlib import Path


# --- Utils ---
# -- Data Working --
def load_data():
    with open(Path(__file__).parent.parent / "data" / "current" / "data.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def update_data(data):
    file_path = Path(__file__).parent.parent / "data" / "current" / "data.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)