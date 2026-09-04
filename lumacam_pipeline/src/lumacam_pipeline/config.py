"""INI configuration helpers for section-based sessions."""

from __future__ import annotations

import ast
import configparser
import json
from pathlib import Path


def _decode(value):
    text = str(value).strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path):
    parser = configparser.ConfigParser()
    read = parser.read(Path(path).expanduser())
    if not read:
        raise FileNotFoundError(f"Configuration file was not found: {path}")
    preset = parser.get("pipeline", "preset", fallback="operando_nci")
    values = {}
    for section in parser.sections():
        if section == "pipeline":
            continue
        values[section] = {name: _decode(value) for name, value in parser.items(section)}
    return preset, values


def save_config(path, preset, values):
    parser = configparser.ConfigParser()
    parser["pipeline"] = {"preset": preset}
    for section, section_values in values.items():
        encoded = {}
        for name, value in section_values.items():
            if isinstance(value, (dict, list, tuple)):
                encoded[name] = json.dumps(value)
            elif value is None:
                encoded[name] = ""
            else:
                encoded[name] = str(value)
        parser[section] = encoded
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return target
