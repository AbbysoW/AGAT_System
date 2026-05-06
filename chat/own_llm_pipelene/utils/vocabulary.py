import json

def load_vocabulary(path: str) -> tuple[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        token_id_V = json.load(f)

    id_token_V = {v: k for k, v in token_id_V.items()}

    return id_token_V, token_id_V