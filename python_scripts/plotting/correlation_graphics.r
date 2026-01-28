library(corrplot)
library(viridisLite)

args <- commandArgs(trailingOnly=TRUE)
numeric_data <- read.csv(args[1])
output_dir <- args[2]

cor_methods <- c("pearson", "spearman", "kendall")
titles <- c("Pearson Correlation", "Spearman Correlation", "Kendall Correlation")

pdf(file.path(output_dir, "correlation_matrices.pdf"), width = 7, height = 7)

for (i in 1:3) {
  method <- cor_methods[i]
  title <- titles[i]
  cor_matrix <- cor(numeric_data, method = method, use = "pairwise.complete.obs")
  
  corrplot.mixed(
    cor_matrix,
    lower = "number",
    upper = "circle",
    tl.pos = "d",         # Diagonal label placement
    tl.col = "black",
    number.cex = 1.5,
    tl.cex = 1.5,
  )
  title(main = title, line = +1, cex.main = 1.5)
}
dev.off()

