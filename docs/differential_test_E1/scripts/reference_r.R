# E1 reference run in R: cases D1, D2, R1, C1.
#
# Purpose: independent implementations (base R lm, class::knn, base
# quantile type 7) of the E1 cases on the same train and test files.
# Input: data/<name>_E1_base.csv, _train.csv, _test.csv from the E1 folder.
# Test: stops if the row counts are not 120/30 and 353/89 or a predictor has NA.
# Output: reference_outputs/r/<case>_r_run<N>.csv and versions_run<N>.txt.
# Run, from the E1 folder, twice (run 1 and run 2 per PLAN.txt controls):
#   Rscript scripts/reference_r.R 1
#   Rscript scripts/reference_r.R 2
# Package needed: class (ships with R). No others.

args <- commandArgs(trailingOnly = TRUE)
run <- if (length(args) >= 1) args[1] else "1"
out <- file.path("reference_outputs", "r")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
options(digits = 17)
fmt <- function(x) formatC(x, digits = 17, format = "g")

writeLines(c(R.version.string, paste("class", as.character(packageVersion("class"))),
             paste("platform", R.version$platform)),
           file.path(out, paste0("versions_run", run, ".txt")))

# D1, D2 descriptive statistics, quantile type 7 (linear), sd with n-1
describe <- function(df, cols) {
  do.call(rbind, lapply(cols, function(cn) {
    s <- df[[cn]]
    q <- quantile(s, c(0.25, 0.5, 0.75), type = 7, names = FALSE)
    data.frame(column = cn, n = sum(!is.na(s)), mean = fmt(mean(s)), sd = fmt(sd(s)),
               min = fmt(min(s)), q1 = fmt(q[1]), median = fmt(q[2]), q3 = fmt(q[3]),
               max = fmt(max(s)), stringsAsFactors = FALSE)
  }))
}
diab_base <- read.csv("data/diabetes_E1_base.csv", check.names = FALSE)
iris_base <- read.csv("data/iris_E1_base.csv", check.names = FALSE)
diab_cols <- c("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target")
iris_cols <- c("sepal length", "sepal width", "petal length", "petal width")
write.csv(describe(diab_base, diab_cols), file.path(out, paste0("D1_r_run", run, ".csv")), row.names = FALSE)
write.csv(describe(iris_base, iris_cols), file.path(out, paste0("D2_r_run", run, ".csv")), row.names = FALSE)

# R1 ordinary least squares on diabetes
tr <- read.csv("data/diabetes_E1_train.csv", check.names = FALSE)
te <- read.csv("data/diabetes_E1_test.csv", check.names = FALSE)
stopifnot(nrow(tr) == 353, nrow(te) == 89, !anyNA(tr[diab_cols]), !anyNA(te[diab_cols]))
X <- diab_cols[diab_cols != "target"]
form <- as.formula(paste("target ~", paste(sprintf("`%s`", X), collapse = " + ")))
fit <- lm(form, data = tr)
pred <- as.numeric(predict(fit, newdata = te))
resid <- te$target - pred
rmse <- sqrt(mean(resid^2))
mae <- mean(abs(resid))
r2 <- 1 - sum(resid^2) / sum((te$target - mean(te$target))^2)
co <- coef(fit)
coef_tab <- data.frame(term = c("intercept", X, "test_rmse", "test_mae", "test_r2"),
                       value = fmt(c(unname(co[1]), unname(co[-1]), rmse, mae, r2)),
                       stringsAsFactors = FALSE)
write.csv(coef_tab, file.path(out, paste0("R1_r_run", run, "_coefficients_metrics.csv")), row.names = FALSE)
write.csv(data.frame(order = te$order, target = te$target, predicted = fmt(pred)),
          file.path(out, paste0("R1_r_run", run, "_predictions.csv")), row.names = FALSE)

# C1 k-nearest neighbours on iris, k = 5, Euclidean, unscaled. class::knn breaks
# ties at random; the seed is set so a rerun is repeatable, and any row whose
# five neighbours tie is a tie-break case, not a defect, in either program.
library(class)
itr <- read.csv("data/iris_E1_train.csv", check.names = FALSE)
ite <- read.csv("data/iris_E1_test.csv", check.names = FALSE)
stopifnot(nrow(itr) == 120, nrow(ite) == 30, !anyNA(itr[iris_cols]), !anyNA(ite[iris_cols]))
set.seed(20260924)
pk <- knn(train = itr[iris_cols], test = ite[iris_cols], cl = factor(itr$species), k = 5)
write.csv(data.frame(order = ite$order, species = ite$species, predicted = as.character(pk)),
          file.path(out, paste0("C1_r_run", run, "_predictions.csv")), row.names = FALSE)
acc <- mean(as.character(pk) == ite$species)
cm <- table(true = ite$species, pred = factor(pk, levels = sort(unique(itr$species))))
writeLines(c(paste("accuracy", fmt(acc)), "confusion rows=true cols=pred", capture.output(print(cm))),
           file.path(out, paste0("C1_r_run", run, "_metrics.txt")))

# C2 is not run in R: glm and nnet::multinom fit unregularized models, which
# PLAN.txt records as "different estimator". Not comparable on equality.
cat("E1 R reference run", run, "written to", out, "\n")
cat("R1 test_rmse", fmt(rmse), " C1 accuracy", fmt(acc), "\n")
