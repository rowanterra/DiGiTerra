"""The example files the welcome screen can load.

An example goes through the ordinary upload, so this route only has to hand the
bytes over and refuse anything not on the list.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from core.flask_app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_the_example_is_served(client):
    r = client.get("/example-data/penguins.csv")
    assert r.status_code == 200
    text = r.data.decode("utf-8")
    assert text.splitlines()[0].startswith("species,island,")


@pytest.mark.parametrize(
    "name",
    ["nothing.csv", "../requirements.txt", "..%2frequirements.txt", "penguins.csv.bak", "app.py"],
)
def test_anything_not_on_the_list_is_refused(client, name):
    assert client.get(f"/example-data/{name}").status_code in (400, 404)


def test_the_shipped_penguins_file_is_the_one_the_docs_describe():
    """The tutorial and examples/README state these numbers; they must hold."""
    d = pd.read_csv("examples/penguins.csv")
    assert d.shape == (344, 7)
    assert list(d.columns) == [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]
    assert d["species"].value_counts().to_dict() == {"Adelie": 152, "Gentoo": 124, "Chinstrap": 68}
    assert int(d.isna().any(axis=1).sum()) == 11
    # The two columns the tutorial says a transform expands.
    assert d["island"].dtype == object and d["sex"].dtype == object


# --- The rows the inference page offers -----------------------------------

NEW_COLUMNS = [
    "row_id",
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex",
]


def test_the_example_rows_are_the_file_the_docs_describe():
    d = pd.read_csv("examples/penguins_new.csv")
    assert len(d) == 22
    assert list(d.columns) == NEW_COLUMNS, list(d.columns)
    assert int(d.isna().sum().sum()) == 0, "the example rows are meant to have no blanks"
    assert set(d["island"]) == {"Biscoe", "Dream", "Torgersen"}, sorted(set(d["island"]))
    assert set(d["sex"]) == {"Female", "Male"}, sorted(set(d["sex"]))


def test_the_key_covers_the_same_rows_with_the_stated_species():
    rows = pd.read_csv("examples/penguins_new.csv")
    key = pd.read_csv("examples/penguins_new_key.csv")
    assert list(key.columns) == ["row_id", "species"]
    assert list(key["row_id"]) == list(rows["row_id"]), "the key does not cover the same rows"
    assert key["species"].value_counts().to_dict() == {"Adelie": 8, "Gentoo": 8, "Chinstrap": 6}


def test_both_example_rows_files_are_served(client):
    for name in ("penguins_new.csv", "penguins_new_key.csv"):
        r = client.get(f"/example-data/{name}")
        assert r.status_code == 200, name
        assert r.data.decode("utf-8").splitlines()[0].startswith("row_id,"), name


def test_the_generator_reproduces_both_files(tmp_path):
    """The rows are synthetic, so the script is the record of how they were made.

    It writes beside itself, so a copy of it and of penguins.csv in a temporary
    directory writes there and leaves the repository alone.
    """
    root = Path.cwd()
    for name in ("make_penguins_new.py", "penguins.csv"):
        shutil.copy(root / "examples" / name, tmp_path / name)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "make_penguins_new.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-800:]
    for name in ("penguins_new.csv", "penguins_new_key.csv"):
        made = (tmp_path / name).read_bytes()
        shipped = (root / "examples" / name).read_bytes()
        assert made == shipped, f"{name} does not reproduce byte for byte"
