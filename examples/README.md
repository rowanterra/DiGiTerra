# Example datasets

Nine CSV files for end-to-end checks of DiGiTerra. The files are in the repository only. The desktop build does not include them.

The files are static copies of scikit-learn datasets ([scikit-learn.org/stable/datasets.html](https://scikit-learn.org/stable/datasets.html), BSD license), except `synthetic_regression.csv`, which was produced once with `sklearn.datasets.make_regression`. No generator script is in the repository.

## Classification

| File | Rows | Features | Target |
|------|------|----------|--------|
| `iris.csv` | 150 | 4 | `species` (setosa, versicolor, virginica) |
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
