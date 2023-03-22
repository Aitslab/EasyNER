# coding=utf-8

import json
import os


def append_to_json_file(path: str, new_data: dict):
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")

    with open(path, "r", encoding="utf-8") as f:
        old_data = json.loads(f.read())

    data = {**old_data, **new_data}  # Merge dicts (new overwrites old)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False))
        
