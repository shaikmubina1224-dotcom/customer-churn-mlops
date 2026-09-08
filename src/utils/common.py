"""
common.py

Common utility functions used throughout the project.
"""

import os
import yaml
import joblib
import json


def read_yaml(file_path: str) -> dict:
    """
    Read a YAML configuration file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def create_directories(paths: list):
    """
    Create directories if they do not already exist.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(file_path: str, obj):
    """
    Save a Python object using joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)


def load_object(file_path: str):
    """
    Load a saved joblib object.
    """
    return joblib.load(file_path)


def save_json(file_path: str, data: dict):
    """
    Save dictionary as JSON.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def load_json(file_path: str):
    """
    Load JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)