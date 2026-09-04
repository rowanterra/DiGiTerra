"""The example files the welcome screen can load.

An example goes through the ordinary upload, so this route only has to hand the
bytes over and refuse anything not on the list.
"""

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
