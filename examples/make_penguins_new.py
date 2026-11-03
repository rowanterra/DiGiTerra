"""Write examples/penguins_new.csv and examples/penguins_new_key.csv.

penguins_new.csv is a synthetic file for the Inference on New Data page. Its
columns are the six indicator columns of penguins.csv, in the same order, with a
row_id in front. No row is a real penguin. Each row is drawn from a multivariate
normal fitted to one species of the complete cases in penguins.csv, so a model
trained on penguins.csv has something to predict, and the key file records the
species each row was drawn from.

Run from the repository root:
    python examples/make_penguins_new.py

Input: examples/penguins.csv with the palmerpenguins columns (species, island,
bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex). Rows with a
blank are dropped before fitting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEED = 33
COUNTS = [("Adelie", 8), ("Chinstrap", 6), ("Gentoo", 8)]
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def main() -> None:
    data = pd.read_csv(ROOT / "penguins.csv").dropna()
    rng = np.random.default_rng(SEED)
    rows = []
    for species, count in COUNTS:
        group = data[data.species == species]
        # Fit one Gaussian per species to the four measurements, then draw.
        draw = rng.multivariate_normal(group[NUMERIC].mean().values, np.cov(group[NUMERIC].values.T), size=count)
        islands = rng.choice(group.island.values, size=count)
        sexes = rng.choice(group.sex.values, size=count)
        for i in range(count):
            rows.append(
                {
                    "island": islands[i],
                    "bill_length_mm": round(float(draw[i, 0]), 1),
                    "bill_depth_mm": round(float(draw[i, 1]), 1),
                    "flipper_length_mm": int(round(draw[i, 2])),
                    "body_mass_g": int(round(draw[i, 3] / 25) * 25),
                    "sex": sexes[i],
                    "species": species,
                }
            )
    new = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    new.insert(0, "row_id", range(1, len(new) + 1))
    # Torgersen holds only Adelie in the source, so two Adelie rows take it and
    # every island in the training file appears in the new file.
    adelie = new.index[new.species == "Adelie"][:2]
    new.loc[adelie, "island"] = "Torgersen"
    assert len(new) == sum(c for _, c in COUNTS)
    new.drop(columns="species").to_csv(ROOT / "penguins_new.csv", index=False)
    new[["row_id", "species"]].to_csv(ROOT / "penguins_new_key.csv", index=False)


if __name__ == "__main__":
    main()
