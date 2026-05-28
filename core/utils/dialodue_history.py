import csv
from pathlib import Path
from datetime import datetime

        # '/hint',
        # '/sys',
        # '/cv',
        # '/user',
        # '/model'

def make_dialogue_history():
    file_path = Path(__file__).parent.parent / "data" / "history" / "dialogue_history" / (str(datetime.now().date()) + ".csv")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        with open(file_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["/user", "/sys", "/cv", "/model", "/hint"])

def add_dialogue_entry(user='', sys='', cv='', model='', hint=''):
    file_path = Path(__file__).parent.parent / "data" / "history" / "dialogue_history" / (str(datetime.now().date()) + ".csv")
    if not file_path.exists():
        make_dialogue_history()
    with open(file_path, 'a', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow([user, sys, cv, model, hint])