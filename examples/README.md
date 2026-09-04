# Example datasets

Ten CSV files for end-to-end checks of DiGiTerra. The files are in the repository only. The desktop build does not include them.

Eight files are static copies of scikit-learn datasets ([scikit-learn.org/stable/datasets.html](https://scikit-learn.org/stable/datasets.html), BSD license). `synthetic_regression.csv` was produced once with `sklearn.datasets.make_regression`. `penguins.csv` is the Palmer Archipelago penguin data of Gorman, Williams and Fraser (2014), distributed under CC0 through the palmerpenguins package and copied here unmodified. No generator script is in the repository.

## Classification

| File | Rows | Features | Target |
|------|------|----------|--------|
| `iris.csv` | 150 | 4 | `species` (setosa, versicolor, virginica) |
| `penguins.csv` | 344 | 6 | `species` (Adelie 152, Gentoo 124, Chinstrap 68) |
| `wine.csv` | 178 | 13 | `class` (class_0, class_1, class_2) |
| `breast_cancer.csv` | 569 | 30 | `diagnosis` (malignant, benign) |

## Regression

| File | Rows | Features | Target |
|------|------|----------|--------|
| `diabetes.csv` | 442 | 10 | `target` |
| `synthetic_regression.csv` | 200 | 10 | `y` |
| `linnerud.csv` | 20 | 3 | `Weight`, `Waist`, `Pulse` (3 targets) |

## Clustering

Feature-only files with no target column. Every column is an indicator.

| File | Rows | Features |
|------|------|----------|
| `iris_clustering.csv` | 150 | 4 (sepal and petal) |
| `wine_clustering.csv` | 178 | 13 |
| `breast_cancer_clustering.csv` | 569 | 30 |

## Procedure

1. Start the app (`python app.py` or the desktop build).
2. Upload the CSV, for example `examples/iris.csv`.
3. On Data Exploration, run the correlation matrix or pairplot.
4. On Model Preprocessing, enter the indicator and target columns as spreadsheet letters, then continue to Modeling.
5. On Modeling, pick the matching task (Classification, Regression, or Clustering) and a model, then train.

## Column setup

| File | Indicators | Target |
|------|------------|--------|
| `iris.csv` | `A-D` | `E` (species) |
| `wine.csv` | `A-M` | `N` (class) |
| `breast_cancer.csv` | `A-AD` | `AE` (diagnosis) |
| `diabetes.csv` | `A-J` | `K` (target) |
| `synthetic_regression.csv` | `A-J` | `K` (y) |
| `linnerud.csv` | `A-C` (Chins, Situps, Jumps) | `D-F` (Weight, Waist, Pulse), multi-target |
| `*_clustering.csv` | all columns | none (choose the Clustering task and a model such as K-Means or GMM) |

Column letters depend on the column order in the file, so the headers of the uploaded file are the reference.

## penguins.csv

The other files are numeric and complete. This one is neither, which is why the
tutorial uses it.

| Column | Letter | Type | Note |
| --- | --- | --- | --- |
| `species` | A | text | The target. Adelie 152, Gentoo 124, Chinstrap 68. |
| `island` | B | text | Three values. A transform makes one indicator column per value. |
| `bill_length_mm` | C | number | 2 values missing. |
| `bill_depth_mm` | D | number | 2 values missing. |
| `flipper_length_mm` | E | number | 2 values missing. |
| `body_mass_g` | F | number | 2 values missing. |
| `sex` | G | text | 11 values missing. Two values otherwise. |

Eleven of the 344 rows have at least one missing value. Entering `B-G` as the
indicators and `A` as the target exercises one-hot encoding, missing-value
handling, a stratified split and three-class classification in one run.
