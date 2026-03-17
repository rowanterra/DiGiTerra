"""Unit tests for python_scripts.config (paths and update_vis_dir)."""
import os
import tempfile
from pathlib import Path


def test_base_dir_is_path_and_contains_python_scripts():
    from python_scripts import config

    assert isinstance(config.BASE_DIR, Path)
    assert (config.BASE_DIR / "python_scripts").is_dir()


def test_log_dir_and_app_support_are_paths():
    from python_scripts import config

    assert isinstance(config.LOG_DIR, Path)
    assert isinstance(config.APP_SUPPORT_DIR, Path)
    assert config.UPLOAD_DIR == config.APP_SUPPORT_DIR / "uploads"


def test_vis_dir_updated_by_update_vis_dir():
    from python_scripts import config

    original_vis = config.VIS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        new_path = Path(tmp) / "custom_vis"
        config.update_vis_dir(new_path)
        try:
            assert config.VIS_DIR == new_path
            assert config.VIS_DIR.exists()
        finally:
            config.update_vis_dir(original_vis)
