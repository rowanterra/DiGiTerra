# DiGiTerra 0.1.0

[![tests](https://github.com/rowanterra/DiGiTerra/actions/workflows/tests.yml/badge.svg)](https://github.com/rowanterra/DiGiTerra/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-forestgreen.svg)](https://github.com/rowanterra/DiGiTerra/blob/main/LICENSE)
![Python 3.11 to 3.12](https://img.shields.io/badge/python-3.11%20to%203.12-forestgreen.svg)

A Flask application for data exploration, preprocessing, model training, and inference on tabular data (CSV, Excel, and Parquet files), with a desktop wrapper for macOS, Windows, and Linux.

## Install and run

From the project root (the directory with `app.py` and `requirements.txt`):

```bash
pip install -r requirements.txt
```

Then run one of the two entry points:

| Goal | Command |
|------|---------|
| Web app in the browser | `python app.py`, then open http://127.0.0.1:5000 |
| The guided page | `python app.py`, then open http://127.0.0.1:5000/guided |
| Desktop window | `python desktop_app.py` |

The guided page is four steps, one on screen at a time with no scrolling, with a Guided and a Full mode and an explanation on every control; Full opens every setting the training route accepts and a hyperparameter table for every model, filled from the estimator that the trainer builds, so the Default column shows the value a blank field leaves. `docs/SIMPLE_INTERFACE_DESIGN.md` says why it looks the way it does, and `docs/guided_page_guided_mode.png` and `docs/guided_page_full_mode.png` show it.

Both entry points import the application from `core/`, `routes/`, and `python_scripts/`.

Tests: `pip install -r requirements-dev.txt` then `MPLBACKEND=Agg pytest tests/ -q`, or `make test`. Production web server: `gunicorn --workers 1 --worker-class gthread --threads 8 app:app` (see `deploy/README.md`). `make dev` installs the Python and npm dependencies, and `make check` runs lint, tests, the bundle check, and the requirements check.

## End-user install without a terminal

The prebuilt application contains the Python interpreter and every dependency, so an end user does not install Python or pip.

### macOS

1. Download `DiGiTerra.app` (or the `.dmg` that contains it) from Releases.
2. Drag `DiGiTerra.app` into `/Applications`.
3. Double-click the app.

### Windows

1. Download the `DiGiTerra` folder from Releases.
2. Extract the folder, for example to `C:\Program Files\DiGiTerra`.
3. Double-click `DiGiTerra.exe`.

### Linux

1. Download the `DiGiTerra` folder from Releases.
2. Extract the folder, for example to `~/Applications/DiGiTerra`.
3. Run `chmod +x DiGiTerra/DiGiTerra`.
4. Double-click `DiGiTerra` or run `./DiGiTerra/DiGiTerra`.

## Example datasets

The `examples/` directory contains 10 CSV files: 4 classification (Iris, Penguins, Wine, Breast cancer), 3 regression (Diabetes, Synthetic, Linnerud), and 3 feature-only clustering copies. Penguins is the one with categorical columns and missing values. `examples/README.md` lists the column setup for each file.

## Verification

`scripts/train_every_model.py` trains every registered estimator on the matching example
dataset through the same function the `/process` route calls, with the defaults an empty
hyperparameter form sends. `docs/every_model_sweep_2026-09-25c.json` is the record of the
run on 2026-09-25 in a Linux container with scikit-learn 1.9.0: 69 of 69 registry entries
trained, Polynomial and Perceptron included (the earlier records of 2026-09-25, 2026-09-24 and
2026-09-03, with 67 of 69, are kept beside it). `docs/artifact_roundtrip_2026-09-25c.json`
is the matching bundle round-trip record: for 69 of 69 estimators the predictions for the
first eight rows were equal before and after the round trip, Birch included.
`docs/differential_test_E1/` is a black-box comparison of the same rows, settings and
splits against Orange, RapidMiner AI Studio and R with plain scikit-learn as the arbiter;
its `RESULT_2026-09-24_partial.txt` names the three defects it found and their fixes.
`scripts/adversarial_check.py` replays every probe of the adversarial review of 2026-09-24
(a crafted artifact, path traversal, a foreign session, formula cells, a CSV with markup in its
headers through both pages, cleared fields, a model-change race, Escape on a focused
tooltip, figures in a 720 px window, and the review's pipeline tests) and prints one line
per probe; `docs/adversarial_check_2026-09-25b.json` is its record at commit 15fe96c, 10 of 10. Run it before
a release and after any change to the routes, the pipeline or the two pages.
`tests/integration/test_browser_advanced.py` drives the page in Chromium through Playwright
and checks that values typed into Advanced-mode controls reach the training request; the
test skips itself when Playwright or the browser is absent.

## Run records and bundles

Every training run writes `manifest.json` and `model_card.html` into the session's output
directory: the SHA-256 of the uploaded file, the request, seed, estimator and parameters,
preprocessing and split, metrics, the baseline comparison, data-check findings, and package
versions. The result page links the model card and `GET /export-bundle`, a zip of the record
with the fitted objects. `scripts/score_bundle.py BUNDLE.zip NEW_ROWS.csv` scores a file from
a bundle without the web app and writes the same columns as the web app, novelty and range included; `scripts/rerun_bundle.py BUNDLE.zip DATA.csv` checks the data
hash, replays the recorded request, and reports whether the metrics reproduce. Loading a
bundle builds fitted objects from a file, so the artifact is written in a format that
stores them as data with their types named and refuses to build a type outside an
exact allowlist (`python_scripts/reporting/artifact_allowlist.txt`, written by
`scripts/artifact_type_census.py` from every registered estimator), and every bundle is signed under the installation's key as well: import refuses
one that is unsigned, altered, or signed elsewhere, and installations that share
`DIGITERRA_BUNDLE_KEY` can exchange bundles. `scripts/check_artifact_roundtrip.py` trains
every registered estimator and compares its predictions before and after the round trip;
`docs/artifact_roundtrip_2026-09-24.json` is the record, with the numbers in it. The in-app
import (`POST /import-bundle`) is on in the desktop build and off in the container unless
`DIGITERRA_ALLOW_BUNDLE_IMPORT` is set.

A fitted model read from a bundle predicts the same values on any machine. Retraining
reproduces a run on the same type of processor. Tree models grown to full depth (random
forest, extra trees, bagging) can resolve exact ties between splits differently on
x86_64 and arm64, because the Apple silicon build of scikit-learn fuses multiplies and
adds, so a retrained RF, ExtraTrees or Bagging model can differ between an Intel machine
and an Apple silicon Mac with the same data, seed and package versions
(`docs/tree_platform_difference_2026-09-25`). The manifest records the operating system
and processor type, `scripts/rerun_bundle.py` prints a note when either differs, and
`scripts/compare_roundtrip_records.py` compares two round-trip records and marks tree
models that differ between processor types.

## Regression stability across seeds

`python_scripts/preprocessing/evaluate_regression_robustness.py` repeats the train/test split and cross-validation over a range of seeds and records the spread:

```bash
python python_scripts/preprocessing/evaluate_regression_robustness.py \
  --input /path/to/data.xlsx \
  --sheet Literature \
  --target TREE \
  --features pH,Fe,Mn,Al,SO4 \
  --keep-feature pH \
  --stratify-col pH \
  --seed-start 0 \
  --seed-count 30 \
  --output-dir output \
  --output-stem tree_robustness
```

The script writes `output/tree_robustness_per_seed.csv` (one row per seed, model, and feature set) and `output/tree_robustness_summary.csv` (median, mean, standard deviation, and 95% empirical interval).

## Building the desktop application

These steps are for maintainers. Run every command from the project root.

1. Install the dependencies and PyInstaller:

```bash
pip install -r requirements.txt
pip install pyinstaller
```

2. Run the build for the platform.

macOS:

```bash
chmod +x build/build_macos.sh
./build/build_macos.sh
```

or `pyinstaller build/DiGiTerra.spec`. The result is `dist/DiGiTerra.app`. macOS needs an `.icns` icon; `build/DiGiTerra.spec` describes how to make one.

Windows:

```cmd
build\build_windows.bat
```

or `pyinstaller build\DiGiTerra_Windows.spec`. The result is the `dist\DiGiTerra` folder with `DiGiTerra.exe`. The spec references `static/Terra_Axe_Logo.ico`.

Linux:

```bash
chmod +x build/build_linux.sh
./build/build_linux.sh
```

or `pyinstaller build/DiGiTerra_Linux.spec`. The result is the `dist/DiGiTerra` folder. Run `chmod +x dist/DiGiTerra/DiGiTerra` before distribution.

## Requirements check

`python scripts/check_requirements.py` imports each package listed in `requirements.txt` and reports the ones that fail to import. The script does not scan the source for imports, so a package that the code needs but `requirements.txt` omits is not detected.

## Configuration

The desktop launcher picks a random open port. `DIGITERRA_PORT` overrides the port:

```bash
export DIGITERRA_PORT=5050
```

On Windows, `set DIGITERRA_PORT=5050`.

If the desktop app closes at once, set `DIGITERRA_DEBUG=1` (`set DIGITERRA_DEBUG=1` on Windows) and read the log file:

| Platform | Log file |
|----------|----------|
| macOS | `~/Library/Logs/DiGiTerra/digiterra.log` |
| Windows | `%APPDATA%\DiGiTerra\Logs\digiterra.log` |
| Linux | `~/.cache/DiGiTerra/logs/digiterra.log` (under `XDG_CACHE_HOME` when set) |

Application data, including generated visualizations:

| Platform | Directory |
|----------|-----------|
| macOS | `~/Library/Application Support/DiGiTerra/user_visualizations/` |
| Windows | `%APPDATA%\DiGiTerra\user_visualizations\` |
| Linux | `~/.local/share/DiGiTerra/user_visualizations/` (under `XDG_DATA_HOME` when set) |

`docs/DEPENDENCIES.md` lists the version pins and the install check.

## Docker and Kubernetes

Docker and Helm assets are under `deploy/`. `deploy/README.md` describes the Docker build and the Helm install.

## Accessibility

The classic page uses ARIA attributes, keyboard navigation with a skip link, screen reader announcements through live regions, and labeled form controls; the Accessibility section of `docs/documentation.md` lists the details. The guided page is built to WCAG 2.2 AA and tested in Chromium with axe-core; `docs/ACCESSIBILITY.md` states what is checked and what is not claimed.

## Acknowledgements

DiGiTerra is built on the scientific Python stack and would not exist without it: scikit-learn (Pedregosa et al., 2011) for every estimator, preprocessing step, and metric the application offers; NumPy (Harris et al., 2020), pandas (McKinney, 2010), and SciPy (Virtanen et al., 2020) for the array, frame, and statistical foundations; matplotlib (Hunter, 2007) and seaborn (Waskom, 2021) for the figures; SHAP (Lundberg & Lee, 2017) for the model explanations; and Flask and Werkzeug (Pallets, 2026) for the web layer. `THIRD_PARTY_NOTICES.md` lists every runtime dependency with its version and license.

Most of the example datasets in `examples/` are static copies of datasets distributed with scikit-learn. Their original sources are the iris measurements of Fisher (1936), the diabetes progression data of Efron et al. (2004), and the wine, breast cancer (WDBC), and Linnerud datasets distributed by the UCI Machine Learning Repository. `synthetic_regression.csv` was generated once with `sklearn.datasets.make_regression`. `penguins.csv` is the Palmer Archipelago penguin data collected by Gorman, Williams and Fraser (2014) and distributed under CC0 through the palmerpenguins package, copied here unmodified. `examples/README.md` gives the column setup for each file.

The Documentation view's styling, which is the evergreen masthead, warm off-white paper, forest-green headings and journal-style tables, follows the comparison page of the corrplotpy package (Terra, 2026).

## References

Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics, 32*(2), 407–499.

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2), 179–188.

Gorman, K. B., Williams, T. D., & Fraser, W. R. (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus *Pygoscelis*). *PLoS ONE, 9*(3), e90081. https://doi.org/10.1371/journal.pone.0090081

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362.

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.

McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61).

Pallets. (2026). *Flask* (Version 3.1.3) [Computer software]. https://flask.palletsprojects.com

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Terra, R. R. (2026). *corrplotpy* (Version 1.0.4) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22287087

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … Vázquez-Baeza, Y. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272.

Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software, 6*(60), 3021.

## License

MIT. See `LICENSE`.
