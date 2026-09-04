# Third-party notices

DiGiTerra is MIT licensed (see `LICENSE`). The application imports the packages below at
runtime, and a PyInstaller build copies them into the distributed application. The
license column was read from each package's installed metadata on 2026-09-04 with
`importlib.metadata`, and from the licence file in the distribution where the metadata
gave no expression; the full license text of each package is in its own distribution.

| Package | Version | License | Source |
| --- | --- | --- | --- |
| flask | 3.1.3 | BSD-3-Clause | https://github.com/pallets/flask/ |
| gunicorn | 23.0.0 | MIT | https://gunicorn.org |
| matplotlib | 3.11.1 | Python Software Foundation License | https://matplotlib.org |
| numpy | 1.26.4 | BSD License | https://numpy.org |
| openpyxl | 3.1.5 | MIT | https://openpyxl.readthedocs.io |
| pandas | 2.3.3 | BSD License | https://pandas.pydata.org |
| scikit-learn | 1.9.0 | BSD-3-Clause | https://scikit-learn.org |
| scipy | 1.17.1 | BSD License | https://scipy.org/ |
| scikit-optimize | 0.10.2 | BSD 3-clause | https://scikit-optimize.readthedocs.io/en/latest/contents.html |
| seaborn | 0.13.2 | BSD License | https://github.com/mwaskom/seaborn |
| shap | 0.49.1 | MIT License | http://github.com/shap/shap |
| werkzeug | 3.1.8 | BSD-3-Clause | https://github.com/pallets/werkzeug/ |
| xlsxwriter | 3.2.9 | BSD-2-Clause | https://github.com/jmcnamara/XlsxWriter |
| pywebview | 5.4 | BSD License | https://pywebview.flowrl.com/ |
| pyarrow | 25.0.1 | Apache-2.0 | https://arrow.apache.org/ |
| prettytable | 3.18.0 | BSD-3-Clause | https://github.com/prettytable/prettytable |
| skops | 0.14.0 | MIT | https://github.com/skops-dev/skops |

The frontend has no runtime JavaScript dependency. ESLint is a development
dependency only (see `package.json`).
