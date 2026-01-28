"""Generate quantile boundary visualization."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from python_scripts.config import VIS_DIR


def quantile_boundary_graphic(train_data, stratifyColumn, quantileNum, pdf_pages=None):
    """
    Create a histogram with quantile boundaries.
    
    Args:
        train_data: DataFrame containing the data
        stratifyColumn: Column name to stratify on
        quantileNum: Number of quantiles
        pdf_pages: PdfPages object (creates new if None, caller responsible for closing)
    """
    created_pdf = False
    if pdf_pages is None:
        pdf_pages = PdfPages(VIS_DIR / "visualizations.pdf")
        created_pdf = True
    
    try:
        quantiles = pd.qcut(train_data[stratifyColumn], q=quantileNum)
        plt.figure(figsize=(10, 6))
        plt.hist(train_data[stratifyColumn], bins=20, color='lightblue', 
                 edgecolor='black', alpha=0.7, label=stratifyColumn)

        for boundary in quantiles.unique():
            plt.axvline(x=boundary.left, color='red', linestyle='--', 
                       label=f'quantile boundary: {boundary}')

        plt.title(f'Distribution of {stratifyColumn} with Quantile Boundaries')
        plt.xlabel(stratifyColumn)
        plt.ylabel('Frequency')
        plt.legend()
        pdf_pages.savefig()
        plt.close()
    finally:
        # Close the PDF if we created it
        if created_pdf:
            pdf_pages.close()



