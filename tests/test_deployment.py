"""Deployment checks: the files the Space reads must agree with each other."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and "==" in line:
            name, ver = line.split("==")
            pins[name.strip().lower()] = ver.strip()
    return pins


def _readme_settings() -> dict[str, str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    assert match, "README.md must begin with the Space settings block"
    settings = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            settings[key.strip()] = value.strip().strip("\"'")
    return settings


def test_sklearn_version_matches_everywhere():
    installed = version("scikit-learn")
    info = json.loads((ROOT / "models" / "model_info.json").read_text())
    assert _requirements()["scikit-learn"] == installed
    assert info["sklearn_version"] == installed


@pytest.mark.parametrize("package", ["gradio", "spaces"])
def test_space_only_packages_not_in_requirements(package):
    assert package not in _requirements()


@pytest.mark.parametrize("package", ["scikit-learn", "pandas", "joblib"])
def test_requirements_are_pinned(package):
    assert package in _requirements(), f"{package} must be pinned with =="


def test_readme_sdk_settings():
    settings = _readme_settings()
    assert settings["sdk"] == "gradio"
    assert settings["sdk_version"] == version("gradio")
    assert settings["python_version"] == "3.12"


def test_app_file_exists():
    assert (ROOT / _readme_settings()["app_file"]).is_file()


def test_model_files_not_ignored():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert not any(line.strip().rstrip("/") == "models" for line in ignored)
    assert (ROOT / "models" / "no_show.joblib").is_file()
