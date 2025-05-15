"""
Author: Sonja Aits (partially based in previous script by Rafsan Ahmed)

"""

# coding=utf-8

import orjson
import os


def append_to_json_file(path: str, new_data: dict):
    if not os.path.isfile(path):
        with open(path, "wb") as f:  # Use binary mode for orjson
            f.write(orjson.dumps({}))

    with open(path, "rb") as f:  # Binary read
        old_data = orjson.loads(f.read())

    data = {**old_data, **new_data}  # Merge dicts (new overwrites old)

    with open(path, "w", encoding="utf-8") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        
