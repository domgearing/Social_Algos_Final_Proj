################################################################################
# Bootstrap polychoric correlation matrices (ordinal; incl. persona vars)
# with optional MICE imputation + stability diagnostics + triangle summary plot
#
# Overview
# --------
# This script constructs and summarizes polychoric correlation matrices from the
# harmonised_data.rds output for a given RATER × YEAR workflow.
#
# We treat *all* variables as ordinal — including persona variables — and:
#   1) Read harmonised data and keep only:
#        persona_id, run, variable, answer
#      assuming exactly one answer per (persona_id, run, variable).
#   2) Cast to wide format (one row per persona_id × run; columns = variables):
#        - If duplicates exist, take the first non-NA per persona/run/variable.
#        - Drop variables that are entirely missing (all NA).
#   3) Convert all non-ID variables to categorical (unordered) for modeling.
#
# Multiple imputation policy (miceRanger)
# --------------------------------------
# If any variables have missing data and run_from_scratch == TRUE, we fit a
# miceRanger multiple-imputation model on the ordinal-variable block:
#   - m = B_MI imputed datasets (mechanisms)
#   - random forest with num.trees = 250, maxiter = 10
#   - single-level variables with missingness are pre-imputed deterministically
#     (fill missing values with the sole observed level) to avoid MI failures.
#
# Bootstrap workflow
# ------------------
# We generate B bootstrap resamples of the wide data (dt_wide). For each bootstrap:
#   - If MI is enabled, we randomly select one of the fitted imputation datasets
#     (1..m) and impute missing values in that bootstrap sample.
#   - We then compute polychoric correlations using:
#        lavaan::lavCor(..., estimator="two.step", missing="pairwise",
#                       cor.smooth=TRUE, se="none")
#   - Variables that are constant (≤ 1 observed non-missing level) within the
#     bootstrap sample are dropped for that bootstrap, since lavCor cannot
#     estimate polychorics for constant ordinals.
#
# Education-stratified correlation matrices
# ----------------------------------------
# In addition to the full-sample polychoric correlation matrix per bootstrap, we
# compute stratified matrices by education category:
#   - Low education:  educ %in% 1:13
#   - High education: educ %in% 14:20
# In each stratum we again drop constant variables and recompute lavCor.
#
# Saved outputs (when run_from_scratch == TRUE)
# ---------------------------------------------
# Writes three lists of correlation matrices (one matrix per bootstrap) to disk:
#   - polychor_bootstrap.rds
#   - polychor_bootstrap_loedu.rds
#   - polychor_bootstrap_hiedu.rds
# If run_from_scratch == FALSE, these are loaded instead of recomputed.
#
# Stability check: raw vs (bootstrap + MI) average correlations (overall only)
# ---------------------------------------------------------------------------
# We compare the polychoric correlation matrix computed on the original (non-
# imputed) data (pairwise missingness) to the *mean* correlation matrix across
# bootstraps after optional MI:
#   - Correlation between matrix entries (upper triangle)
#   - Mean/median/max absolute difference
# We also build pairwise diagnostics linking |Δr| to the mean % missingness
# across the two variables in each pair and define ggplots for:
#   - raw r vs avg-imputed r
#   - mean % missing vs |Δr|
#
# Missingness “airplane” plot (bootstrap distribution)
# ----------------------------------------------------
# Separately, we bootstrap the *raw* wide data (with NAs) and compute the
# distribution of % missing per variable across bootstraps, summarizing with:
#   - median missingness
#   - 2.5% and 97.5% quantiles
# and define a pointrange + coord_flip plot for inspection.
#
# Triangle summary plot: median correlation + Pr(sign)
# ----------------------------------------------------
# We summarize bootstrap correlation matrices into a single triangle plot:
#   - Lower triangle: median correlation per pair across bootstraps (r_med)
#   - Upper triangle: Pr(sign) = proportion of bootstrap correlations having the
#     same sign as the median (blanked if < 0.5)
#
# Implementation details:
#   - stack_corr_boot_long(): stacks bootstrap matrices to long form and
#     canonicalizes (var1,var2) using a global ordering (vars).
#   - summarise_corr_boot_median(): computes r_med, pr_sign, n, n_boot per pair.
#   - plot_corr_triangle_from_summary(): renders the dual-triangle heatmap with
#     separate fill scales (ggnewscale).
#
# Prerequisites (provided elsewhere in the project)
# ------------------------------------------------
# Required globals (typical):
#   - BASE_OUT_DIR, DIR_OUT, BASE_VIZ_DIR
#   - RATER, YEAR
#   - B (number of bootstraps), B_MI (MI datasets), run_from_scratch (TRUE/FALSE)
#
# Required packages
# -----------------
#   - data.table
#   - miceRanger
#   - lavaan
#   - ggplot2
#   - ggnewscale
#   - scales
#   - parallel
#   - (optional for saving PDFs) grDevices / cairo
#
# Output
# ------
# Correlation list RDS files (when recomputed) in DIR_OUT, and optional PDFs in:
#   file.path(BASE_VIZ_DIR, paste0(RATER, "-", YEAR), ...)
################################################################################



################################################################################
# A) LOAD + RESHAPE HARMONISED DATA (long -> wide)
################################################################################
data_j <- readRDS(file = paste0(DIR_OUT,'/harmonised_data.rds'))

# Use everything in data_j, treating all variables as ordinal (including persona vars)
# Keep only the columns we need; assume exactly one answer per (persona_id, run, variable)
dt_long <- as.data.table(data_j)[, .(persona_id, run, variable, answer)]

# Cast to wide: one row per persona_id x run, columns = variables
# If duplicates exist within a persona/run/variable, take the first non-NA
setorder(dt_long, persona_id, run, variable)
dt_wide <- dcast(
  dt_long[!is.na(variable)],
  persona_id + run ~ variable,
  value.var = "answer",
  fun.aggregate = function(x) x[which.max(!is.na(x))],
  fill = NA_real_,
  drop = FALSE
)

# if there are any fully missing variables, drop 
drop.na <- apply(dt_wide,2,function(x){all(is.na(x))})
if(any(drop.na)){
  dt_wide <- dt_wide[,!drop.na,with = F]
}

# Names of the ordinal variables (all except ids)
ord_vars <- setdiff(names(dt_wide), c("persona_id", "run"))

# dt_wide <- to_ordered(dt_wide,ord_vars)
dt_wide <- to_unordered(dt_wide,ord_vars)

# Which vars need imputation? (miceRanger will ignore fully observed vars)
vars_to_impute <- names(dt_wide)[colSums(is.na(dt_wide)) > 0]



################################################################################
# B) MULTIPLE IMPUTATION MODEL (miceRanger)
################################################################################
# Policy:
# - Only fit MI when run_from_scratch == TRUE AND there is at least one variable
#   with missingness.
# - Single-level variables with missingness are pre-imputed deterministically:
#     fill missing with the sole observed level (avoids MI failures).
################################################################################

## --- Fit miceRanger model (only if there is something to impute) ----
if(run_from_scratch == TRUE){
  
  if (length(vars_to_impute) != 0L) {
    
    # --- Handle single-level variables with missing values
    tmp <- dt_wide[, ..ord_vars]
    
    single_level_vars <- names(tmp)[sapply(tmp, function(x) {
      ux <- unique(x[!is.na(x)])
      length(ux) == 1L && anyNA(x)
    })]
    
    if (length(single_level_vars) > 0L) {
      message("Pre-imputing single-level variables: ", paste(single_level_vars, collapse = ", "))
      for (v in single_level_vars) {
        val <- dt_wide[!is.na(get(v)), get(v)][1L]  # the single observed level
        dt_wide[is.na(get(v)), (v) := val]
      }
    } else {
      message("No single-level variables with missing values found.")
    }
    
    # --- Train miceRanger MI model
    mr <- miceRanger::miceRanger(
      data           = dt_wide[, ..ord_vars],
      m              = B_MI,     # number of MI datasets
      maxiter        = 10,
      returnModels   = TRUE,
      num.trees      = 250,
      verbose        = TRUE,
      num.threads    = parallel::detectCores()     
    )
    
  }
  
  
  
  ################################################################################
  # C) BOOTSTRAP RESAMPLING + BOOTSTRAP-LEVEL IMPUTATION
  ################################################################################
  # Bootstrap unit: rows of dt_wide (persona_id × run records).
  # If MI is enabled:
  #   - For each bootstrap replicate, randomly select 1 of the fitted MI datasets
  #     and impute missing values in that bootstrap sample.
  ################################################################################
  
  ## --- Generate bootstrap samples --------------------------------------
  boot_list <- bootstrap(wide_dt = dt_wide, B = B)
  
  ## --- Complete imputation on bootstrap samples -------------
  if (length(vars_to_impute) != 0L) {
    
    for (j in seq_along(boot_list)) {
      # pick a random imputation mechanism (1..m) 
      ds_idx <- sample.int(mr$callParams$m, 1L)
      
      imp_res <- miceRanger::impute(
        boot_list[[j]],
        miceObj  = mr,
        datasets = ds_idx
      )
      
      message("Imputed NAs for bootstrap ", j, "/", B)
      
      # overwrite in place
      boot_list[[j]] <- imp_res$imputedData[[1L]]
    }
  }
  
  
  
  ################################################################################
  # D) COMPUTE POLYCHORIC CORRELATIONS (overall + education-stratified)
  ################################################################################
  # For each bootstrap replicate:
  #   1) Drop variables that are constant within the replicate (<= 1 non-missing level)
  #   2) Convert remaining variables to ordered factors
  #   3) Compute polychoric correlation via lavaan::lavCor (two-step estimator)
  #
  # Education strata (requires `educ` in ord_vars):
  #   - Low education:  educ %in% 1:13
  #   - High education: educ %in% 14:20
  # In each stratum, we again drop constants and recompute correlations.
  ################################################################################
  
  # now caclulate polychoric correrlations
  corr_list <- list()
  corr_lo.edu_list <- list()
  corr_hi.edu_list <- list()
  
  for(j in seq_along(boot_list)){
    
    dat_j <- as.data.table(boot_list[[j]])[, ..ord_vars]
    
    ## 1. Find vars with at least 2 non-missing levels
    keep_vars <- names(dat_j)[
      vapply(dat_j, function(x) length(unique(na.omit(x))) > 1L, logical(1))
    ]
    drop_vars <- setdiff(ord_vars, keep_vars)
    
    if (length(drop_vars)) {
      message("Bootstrap ", j, ": dropping constant vars: ",
              paste(drop_vars, collapse = ", "))
    }
    
    ## 2. Convert only those to ordered
    dat_ord <- to_ordered(dat_j[, ..keep_vars], keep_vars)
    
    corr_list[[j]] <-
      lavaan::lavCor(
        dat_ord,
        ordered     = keep_vars,
        estimator   = "two.step",
        se          = "none",
        output      = "cor",
        cor.smooth  = TRUE,
        missing     = "pairwise"
      )
    
    dat_ord_loedu <- dat_ord[educ %in% 1:13][,!'educ']
    
    keep_vars <- names(dat_ord_loedu)[
      vapply(dat_ord_loedu, function(x) length(unique(na.omit(x))) > 1L, logical(1))
    ]
    drop_vars <- setdiff(ord_vars, keep_vars)
    
    if (length(drop_vars)) {
      message("Bootstrap ", j, ": dropping constant vars: ",
              paste(drop_vars, collapse = ", "))
    }
    
    dat_ord_loedu <- to_ordered(dat_ord_loedu[, ..keep_vars], keep_vars)
    
    corr_lo.edu_list[[j]] <-
      lavaan::lavCor(
        dat_ord_loedu, # some college is threshold 
        ordered     = keep_vars,
        estimator   = "two.step",
        se          = "none",
        output      = "cor",
        cor.smooth  = TRUE,
        missing     = "pairwise"
      )
    
    
    dat_ord_hiedu <- dat_ord[educ %in% 14:20][,!'educ']
    
    keep_vars <- names(dat_ord_hiedu)[
      vapply(dat_ord_hiedu, function(x) length(unique(na.omit(x))) > 1L, logical(1))
    ]
    drop_vars <- setdiff(ord_vars, keep_vars)
    
    if (length(drop_vars)) {
      message("Bootstrap ", j, ": dropping constant vars: ",
              paste(drop_vars, collapse = ", "))
    }
    
    dat_ord_hiedu <- to_ordered(dat_ord_hiedu[, ..keep_vars], keep_vars)
    
    corr_hi.edu_list[[j]] <-
      lavaan::lavCor(
        dat_ord_hiedu, # some college is threshold 
        ordered     = keep_vars,
        estimator   = "two.step",
        se          = "none",
        output      = "cor",
        cor.smooth  = TRUE,
        missing     = "pairwise"
      )
  }
  
  # save the correlation matrices 
  saveRDS(corr_list, file = file.path(DIR_OUT, "polychor_bootstrap.rds"))
  saveRDS(corr_lo.edu_list, file = file.path(DIR_OUT, "polychor_bootstrap_loedu.rds"))
  saveRDS(corr_hi.edu_list, file = file.path(DIR_OUT, "polychor_bootstrap_hiedu.rds"))
  
}else{
  corr_list <- readRDS(file = file.path(DIR_OUT, "polychor_bootstrap.rds"))
  corr_lo.edu_list <- readRDS(file = file.path(DIR_OUT, "polychor_bootstrap_loedu.rds"))
  corr_hi.edu_list <- readRDS(file = file.path(DIR_OUT, "polychor_bootstrap_hiedu.rds"))
}



################################################################################
# E) STABILITY CHECK: RAW (PAIRWISE) VS BOOTSTRAP(+MI) AVERAGE (overall only)
################################################################################
# Goal:
# - Compare the original-data polychoric matrix (pairwise missingness; no MI)
#   against the mean of the bootstrap matrices (after optional MI).
# Outputs:
# - Summary stats printed to console
# - A pairwise table linking |Δr| to missingness
# - Two ggplots saved as a 1×2 PDF
# - Correlation between missingness and |Δr| saved as RDS
################################################################################

## --- Compare polychoric correlations: before vs after MICE (only overall) ----

# 1. Polychoric correlations on original (non-imputed) data
dat_orig <- as.data.table(dt_wide)[, ..ord_vars]

# Drop ordered vars that have only 1 non-missing level (lavaan cannot handle them)
const_vars <- names(dat_orig)[
  vapply(dat_orig, function(x) length(unique(na.omit(x))) <= 1L, logical(1))
]

if (length(const_vars)) {
  message("Dropping constant vars from original data: ",
          paste(const_vars, collapse = ", "))
}

ord_vars_c <- setdiff(ord_vars, const_vars)
dat_orig_c <- dat_orig[, ..ord_vars_c]

cor_orig <- lavaan::lavCor(
  dat_orig_c,
  ordered    = ord_vars_c,
  estimator  = "two.step",
  se         = "none",
  output     = "cor",
  cor.smooth = TRUE,
  missing    = "pairwise"
)

# 2. Polychoric correlations on average across bootstraps
corr_b_mean <- mean_corr_pairwise(corr_list)

# 3. Align to common set of variables
common_vars <- intersect(rownames(cor_orig), rownames(corr_b_mean))

cor_orig_c <- cor_orig[common_vars, common_vars, drop = FALSE]
corr_b_mean_c  <- corr_b_mean[common_vars, common_vars, drop = FALSE]

# 4. Compare matrices: upper triangular elements
get_ut <- function(m) m[upper.tri(m, diag = FALSE)]

v_orig <- get_ut(cor_orig_c)
v_imp  <- get_ut(corr_b_mean_c)

# Correlation between the two sets of correlation coefficients
cor_between_mats <- cor(v_orig, v_imp)

# Absolute differences
diff_abs <- abs(v_imp - v_orig)

cat("\n--- Comparison of polychoric correlation matrices ---\n")
cat("Correlation between original and avg. imputed correlation entries: ",
    round(cor_between_mats, 3), "\n")
cat("Mean absolute difference: ", round(mean(diff_abs), 4), "\n")
cat("Median absolute difference: ", round(median(diff_abs), 4), "\n")
cat("Max absolute difference: ", round(max(diff_abs), 4), "\n\n")

## --- 0. Restrict original data to the common vars -----------------------
dat_orig_c <- as.data.table(dat_orig)[, ..common_vars]

## --- 1. % missing per variable in original data -------------------------
miss_pct <- dat_orig_c[, lapply(.SD, function(x) mean(is.na(x)) * 100)]
miss_pct <- as.numeric(miss_pct[1, ])
names(miss_pct) <- common_vars
# miss_pct is now a named numeric vector: var -> % missing

## --- 2. Build full pairwise table of correlation differences ------------

# indices of upper triangle
ut_idx <- which(upper.tri(cor_orig_c, diag = FALSE), arr.ind = TRUE)

# variable names for each pair
var1 <- rownames(cor_orig_c)[ut_idx[, "row"]]
var2 <- colnames(cor_orig_c)[ut_idx[, "col"]]

# original & imputed correlations for those pairs
cor_orig_ut <- cor_orig_c[upper.tri(cor_orig_c, diag = FALSE)]
cor_imp_ut  <- corr_b_mean_c[upper.tri(corr_b_mean_c, diag = FALSE)]

# differences
diff_vec     <- cor_imp_ut - cor_orig_ut
diff_abs_vec <- abs(diff_vec)

# assemble into a data.table
pair_diff_dt <- data.table(
  var1          = var1,
  var2          = var2,
  cor_orig      = cor_orig_ut,
  cor_imp       = cor_imp_ut,
  diff          = diff_vec,
  diff_abs      = diff_abs_vec,
  miss_var1_pct = miss_pct[var1],
  miss_var2_pct = miss_pct[var2]
)

# useful composite measures of missingness per pair
pair_diff_dt[, `:=`(
  miss_mean_pct = (miss_var1_pct + miss_var2_pct) / 2,
  miss_max_pct  = pmax(miss_var1_pct, miss_var2_pct)
)]

## --- 3. Rank pairs by absolute difference (if you want to inspect later) --
setorder(pair_diff_dt, -diff_abs)

## --- 4. Stats for Plot 1: v_orig vs v_imp ----------------------------------

# data for plot 1
corr_comp_dt <- data.table(
  cor_orig = pair_diff_dt$cor_orig,
  cor_imp  = pair_diff_dt$cor_imp,
  diff     = pair_diff_dt$diff,
  diff_abs = pair_diff_dt$diff_abs
)

cor_between_mats <- cor(corr_comp_dt$cor_orig, corr_comp_dt$cor_imp,
                        use = "pairwise.complete.obs")
bias <- mean(corr_comp_dt$diff, na.rm = TRUE)       # mean(v_imp - v_orig)
mae  <- mean(corr_comp_dt$diff_abs, na.rm = TRUE)   # mean |v_imp - v_orig|

legend_title_p1 <- paste0(
  "r = ", round(cor_between_mats, 3),
  "\nBias = ", round(bias, 3),
  "\nMAE = ", round(mae, 3)
)

## --- 5. Stats for Plot 2: missingness vs |difference| ----------------------

r_miss_diff <- cor(
  pair_diff_dt$miss_mean_pct,
  pair_diff_dt$diff_abs,
  use = "pairwise.complete.obs"
)

# save this for later analysis
saveRDS(r_miss_diff, file = file.path(DIR_OUT, "corr_mice_stability_check.rds"))


legend_title_p2 <- paste0(
  "r = ", round(r_miss_diff, 3)
)

## --- 6. Build the two ggplots ---------------------------------------------

# Plot 1: v_imp (x) vs v_orig (y)
p1 <- ggplot(corr_comp_dt, aes(x = cor_imp, y = cor_orig)) +
  geom_point(aes(color = "variable pairs"), alpha = 0.5) +
  geom_smooth(aes(color = "loess curve"),
              method = "loess",
              se = TRUE) +
  # optional identity line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  scale_color_manual(
    name   = legend_title_p1,
    values = c("variable pairs" = "grey40", "loess curve" = "blue")
  ) +
  labs(
    x = "imputed + bootstrap avg. corr.",
    y = "raw corr.",
    title = paste0("Polychoric correlations\nraw ", RATER,
                   " vs. bootstrap + MICE avg.")
  ) +
  theme_minimal() +
  theme(
    legend.position      = c(0.05, 0.95),
    legend.justification = c(0, 1)
  )

# Plot 2: |difference| vs mean % missing
p2 <- ggplot(pair_diff_dt, aes(x = miss_mean_pct, y = diff_abs)) +
  geom_point(aes(color = "variable pairs"), alpha = 0.5) +
  geom_smooth(aes(color = "loess curve"),
              method = "loess",
              se = TRUE) +
  scale_color_manual(
    name   = legend_title_p2,
    values = c("variable pairs" = "grey40", "loess curve" = "blue")
  ) +
  labs(
    x = "Mean % missing (pairwise)",
    y = "Absolute difference in corr.",
    title = "Abs. Difference vs Missingness"
  ) +
  theme_minimal() +
  theme(
    legend.position      = c(0.05, 0.95),
    legend.justification = c(0, 1)
  )

## --- 7. Save as 1×2 PDF ---------------------------------------------------

file_path <- file.path(BASE_VIZ_DIR, paste0(RATER,"-", as.character(YEAR)))
dir.create(file_path, recursive = TRUE, showWarnings = FALSE)

pdf(file.path(file_path, "corr_mice_stability_check.pdf"),
    width = 12.5, height = 5)

grid.arrange(p1, p2, ncol = 2)

dev.off()


################################################################################
# F) MISSINGNESS “AIRPLANE” PLOT: BOOTSTRAP DISTRIBUTION OF % MISSING
################################################################################
# Goal:
# - Quantify variability in per-variable missingness under row-resampling.
# - Summarise % missing across bootstraps by median and 95% interval.
# Output:
# - missingness_interval_bootstrap.pdf
################################################################################

## --- Airplane plot: bootstrap distribution of % missing per variable ---

# Recreate bootstrap samples from the *raw* wide data (with NAs)
boot_list_miss <- bootstrap(wide_dt = dt_wide, B = B)

# % missing per variable, per bootstrap
miss_boot_dt <- data.table::rbindlist(
  lapply(seq_along(boot_list_miss), function(j) {
    dat_j <- data.table::as.data.table(boot_list_miss[[j]])[, ..ord_vars]
    data.table::data.table(
      boot     = j,
      variable = names(dat_j),
      miss_pct = colMeans(is.na(dat_j)) * 100
    )
  }),
  use.names = TRUE, fill = TRUE
)

# Median + 95% interval across bootstraps
miss_sum_dt <- miss_boot_dt[, .(
  miss_median = median(miss_pct, na.rm = TRUE),
  miss_lo     = as.numeric(stats::quantile(miss_pct, 0.025, na.rm = TRUE)),
  miss_hi     = as.numeric(stats::quantile(miss_pct, 0.975, na.rm = TRUE))
), by = variable]

# Order by median % missing (highest at top)
data.table::setorder(miss_sum_dt, -miss_median)
miss_sum_dt[, variable := factor(variable, levels = variable)]

p_miss <- ggplot(miss_sum_dt, aes(
  x = variable, y = miss_median, ymin = miss_lo, ymax = miss_hi
)) +
  geom_pointrange() +
  coord_flip() +
  labs(
    x = NULL,
    y = "% missing (bootstrap median, 95% interval)",
    title = "Missingness by variable (bootstrap distribution)"
  ) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 6))

# Save separately
miss_dir <- file.path(BASE_VIZ_DIR, paste0(RATER, "-", as.character(YEAR)))
dir.create(miss_dir, recursive = TRUE, showWarnings = FALSE)

pdf(file.path(miss_dir, "missingness_interval_bootstrap.pdf"),
    width = 8.5, height = max(6, 0.18 * nrow(miss_sum_dt) + 2))
print(p_miss)
dev.off()



################################################################################
# G) BOOTSTRAP CORRELATION SUMMARY + TRIANGLE PLOTTING
################################################################################
# This block:
#  1) stacks a list of bootstrap correlation matrices to long format
#  2) canonicalizes (var1,var2) using global order `vars`
#  3) summarises per-pair median correlation + Pr(sign)
#  4) plots a triangle: lower = median corr, upper = Pr(sign)
#
# Notes for reviewers:
# - Bootstrap matrices may drop variables (constants) within some resamples.
# - Canonicalization is critical so that (A,B) and (B,A) are treated as the same
#   pair when stacking across bootstraps.
################################################################################


#' Stack bootstrap correlation matrices into long format
#'
#' @description
#' Converts a list of correlation matrices (one per bootstrap replicate) into a
#' single long `data.table` with columns: `boot`, `var1`, `var2`, `r`.
#'
#' @details
#' - Only lower-triangular cells are extracted (optionally including diagonal),
#'   to avoid duplicate symmetric entries.
#' - Pairs are *canonicalized* using the global ordering `vars` so that the same
#'   variable pair has the same `(var1, var2)` orientation across bootstraps,
#'   even if matrix row/column order differs.
#' - Pairs where either variable is not in `vars` are dropped.
#'
#' @param corr_list List of correlation matrices (each with row/col names).
#' @param vars Character vector defining the global variable universe and axis order.
#' @param diag Logical; include diagonal entries (default TRUE).
#' @param keep_na_r Logical; keep NA correlations (default FALSE).
#'
#' @return A `data.table` with columns `boot`, `var1`, `var2`, `r`.
#'
#' @examples
#' # DT <- stack_corr_boot_long(corr_list, vars = ord_vars, diag = TRUE)
stack_corr_boot_long <- function(corr_list,
                                 vars,
                                 diag = TRUE,
                                 keep_na_r = FALSE) {
  
  DT <- data.table::rbindlist(
    lapply(seq_along(corr_list), function(b) {
      
      M <- corr_list[[b]]
      if (is.null(M) || !is.matrix(M)) return(NULL)
      if (is.null(rownames(M)) || is.null(colnames(M))) return(NULL)
      
      idx <- which(lower.tri(M, diag = diag), arr.ind = TRUE)
      
      dt_b <- data.table::data.table(
        boot = b,
        var1 = rownames(M)[idx[, 1]],
        var2 = colnames(M)[idx[, 2]],
        r    = as.numeric(M[idx])
      )
      
      # Keep only pairs where both vars are in the global universe
      dt_b <- dt_b[var1 %in% vars & var2 %in% vars]
      if (nrow(dt_b) == 0L) return(NULL)
      
      # Canonicalize orientation using global ordering:
      # ensure match(var1, vars) >= match(var2, vars)
      dt_b[, `:=`(
        p1 = match(var1, vars),
        p2 = match(var2, vars)
      )]
      
      # Safe swap (simultaneous assignment)
      dt_b[p1 < p2, c("var1", "var2") := .(var2, var1)]
      
      dt_b[, c("p1", "p2") := NULL]
      
      if (!keep_na_r) dt_b <- dt_b[!is.na(r)]
      
      dt_b
    }),
    use.names = TRUE,
    fill      = TRUE
  )
  
  if (is.null(DT) || nrow(DT) == 0L) {
    return(data.table::data.table(
      boot = integer(), var1 = character(), var2 = character(), r = numeric()
    ))
  }
  
  DT[]
}





#' Count how often each correlation pair appears across bootstraps
#'
#' @description
#' For each `(var1, var2)` pair in a long correlation table, counts:
#' - how many distinct bootstraps contribute that pair (`n_boot`)
#' - how many rows exist for the pair (`n_rows`)
#' - the fraction of all bootstraps in which the pair appears (`frac_boot`)
#'
#' @param DT A `data.table` produced by `stack_corr_boot_long()`.
#' @param B Optional integer; total number of bootstraps. If NULL, uses `max(DT$boot)`.
#'
#' @return A `data.table` with columns `var1`, `var2`, `n_boot`, `n_rows`, `frac_boot`.
#'
#' @examples
#' # counts <- pair_appearance_counts(DT_long, B = B)
pair_appearance_counts <- function(DT, B = NULL) {
  
  if (is.null(B)) B <- max(DT$boot, na.rm = TRUE)
  
  out <- DT[, .(
    n_boot    = data.table::uniqueN(boot),
    n_rows    = .N,
    frac_boot = data.table::uniqueN(boot) / B
  ), by = .(var1, var2)]
  
  out[order(-n_boot, var1, var2)]
}





#' Summarise bootstrap correlations by median and sign stability
#'
#' @description
#' Computes per-variable-pair summary statistics across bootstraps:
#' - `r_med`: median correlation across bootstraps
#' - `pr_sign`: probability the correlation has the same sign as the median
#' - `n`: number of non-NA correlation values
#' - `n_boot`: number of bootstraps contributing the pair (non-NA)
#'
#' @param corr_list List of bootstrap correlation matrices (one per bootstrap).
#' @param vars Character vector defining the global variable universe and axis order.
#' @param return_long Logical; if TRUE returns both the long table and summary.
#' @param diag Logical; include diagonal entries when stacking (default TRUE).
#' @param keep_na_r Logical; keep NA correlations when stacking (default FALSE).
#'
#' @return If `return_long = FALSE`: a summary `data.table`.
#' If `return_long = TRUE`: a list with `long` and `summary`.
#'
#' @examples
#' # res <- summarise_corr_boot_median(corr_list, ord_vars, return_long = TRUE)
summarise_corr_boot_median <- function(corr_list,
                                       vars,
                                       return_long = FALSE,
                                       diag = TRUE,
                                       keep_na_r = FALSE) {
  
  DT <- stack_corr_boot_long(
    corr_list = corr_list,
    vars      = vars,
    diag      = diag,
    keep_na_r = keep_na_r
  )
  
  if (nrow(DT) == 0L) {
    empty_sum <- data.table::data.table(
      var1 = character(), var2 = character(),
      r_med = numeric(), pr_sign = numeric(), n = integer(), n_boot = integer()
    )
    if (return_long) return(list(long = DT, summary = empty_sum))
    return(empty_sum)
  }
  
  out <- DT[!is.na(r), .(
    r_med = median(r, na.rm = TRUE),
    
    pr_sign = {
      med <- median(r, na.rm = TRUE)
      if (is.na(med)) NA_real_
      else if (med >= 0) mean(r >= 0, na.rm = TRUE)
      else mean(r <= 0, na.rm = TRUE)
    },
    
    n      = sum(!is.na(r)),
    n_boot = data.table::uniqueN(boot)
  ), by = .(var1, var2)]
  
  if (return_long) list(long = DT, summary = out[])
  else out[]
}





#' Plot a triangle heatmap from bootstrap correlation summaries
#'
#' @description
#' Produces a dual-triangle plot:
#' - Lower triangle shows median correlation (`r_med`)
#' - Upper triangle shows Pr(sign) (`pr_sign`) with values < 0.5 blanked
#'
#' @details
#' Uses `ggnewscale::new_scale_fill()` to maintain independent fill scales for
#' the lower and upper triangles.
#'
#' @param sum_dt A summary `data.table` from `summarise_corr_boot_median()`.
#' @param vars Character vector defining axis ordering.
#' @param out_file Optional file path; if provided, saves a PDF via `ggsave()`.
#'
#' @return A `ggplot` object.
#'
#' @examples
#' # p <- plot_corr_triangle_from_summary(sum_dt, ord_vars)
plot_corr_triangle_from_summary <- function(sum_dt, vars, out_file = NULL) {
  
  # Full grid of cells
  grid_dt <- data.table::CJ(var_row = vars, var_col = vars)
  grid_dt[, `:=`(
    i = match(var_row, vars),
    j = match(var_col, vars)
  )]
  
  # Canonical join keys: (key1,key2) always in lower-tri orientation (i>=j)
  grid_dt[, `:=`(
    key1 = ifelse(i >= j, var_row, var_col),
    key2 = ifelse(i >= j, var_col, var_row)
  )]
  
  # Merge summary stats onto grid
  grid_dt <- merge(
    grid_dt,
    sum_dt,
    by.x = c("key1", "key2"),
    by.y = c("var1", "var2"),
    all.x = TRUE
  )
  
  # Axis ordering
  grid_dt[, x := factor(var_col, levels = vars)]
  grid_dt[, y := factor(var_row, levels = rev(vars))]
  
  lower_dt <- grid_dt[i >= j]   # r_med
  upper_dt <- grid_dt[i <  j]   # pr_sign
  
  # Optional: only show pr_sign in [0.5,1]
  upper_dt[!is.na(pr_sign) & pr_sign < 0.5, pr_sign := NA_real_]
  
  p <- ggplot2::ggplot() +
    
    # Lower: median corr
    ggplot2::geom_tile(
      data = lower_dt,
      ggplot2::aes(x = x, y = y, fill = r_med),
      color     = "grey80",
      linewidth = 0.25
    ) +
    ggplot2::scale_fill_gradient2(
      low      = "orangered",
      mid      = "white",
      high     = "springgreen",
      midpoint = 0,
      limits   = c(-1, 1),
      na.value = "grey95",
      name     = "Median corr."
    ) +
    
    ggnewscale::new_scale_fill() +
    
    # Upper: Pr(sign)
    ggplot2::geom_tile(
      data = upper_dt,
      ggplot2::aes(x = x, y = y, fill = pr_sign),
      color     = "grey80",
      linewidth = 0.25
    ) +
    ggplot2::scale_fill_gradient(
      low      = "white",
      high     = "dodgerblue",
      limits   = c(0.5, 1),
      breaks   = seq(0.5, 1, by = 0.1),
      labels   = scales::number_format(accuracy = 0.01),
      na.value = "transparent",
      name     = "Pr(sign)"
    ) +
    
    ggplot2::coord_fixed() +
    ggplot2::labs(x = NULL, y = NULL) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      panel.grid  = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(angle = 90, hjust = 1, vjust = 0.5),
      axis.text.y = ggplot2::element_text(size = 8)
    )
  
  if (!is.null(out_file)) {
    dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(out_file, p, width = 12 * 0.9, height = 10 * 0.8, device = cairo_pdf)
  }
  
  p
}




#' Wrapper: stack -> summarise -> plot triangle heatmap
#'
#' @description
#' Convenience function that summarises a bootstrap correlation list using the
#' median/Pr(sign) summary and then plots the triangle heatmap.
#'
#' @param corr_list List of bootstrap correlation matrices.
#' @param vars Character vector defining axis ordering.
#' @param out_file Optional file path for saving the plot as PDF.
#' @param diag Logical; include diagonal entries during stacking (default TRUE).
#' @param keep_na_r Logical; keep NA correlations during stacking (default FALSE).
#' @param return_long Logical; if TRUE returns plot + long + summary.
#'
#' @return If `return_long = FALSE`: a `ggplot` object.
#' If `return_long = TRUE`: a list with `plot`, `long`, and `summary`.
#'
#' @examples
#' # out <- plot_corr_triangle_median(corr_list, ord_vars, return_long = TRUE)
plot_corr_triangle_median <- function(corr_list, vars, out_file = NULL,
                                      diag = TRUE, keep_na_r = FALSE,
                                      return_long = FALSE) {
  
  res <- summarise_corr_boot_median(
    corr_list   = corr_list,
    vars        = vars,
    return_long = TRUE,
    diag        = diag,
    keep_na_r   = keep_na_r
  )
  
  p <- plot_corr_triangle_from_summary(
    sum_dt   = res$summary,
    vars     = vars,
    out_file = out_file
  )
  
  if (return_long) list(plot = p, long = res$long, summary = res$summary)
  else p
}





################################################################################
# H) RUN: SUMMARISE + PLOT TRIANGLE
################################################################################

res <- summarise_corr_boot_median(
  corr_list    = corr_list,
  vars         = ord_vars,
  return_long  = TRUE,
  diag         = TRUE,
  keep_na_r    = FALSE
)

DT_long <- res$long
sum_dt  <- res$summary

# Optional sanity check: canonical orientation holds
stopifnot(all(match(DT_long$var1, ord_vars) >= match(DT_long$var2, ord_vars)))

out_dir  <- file.path(BASE_VIZ_DIR, paste0(RATER, "-", as.character(YEAR)))
out_file <- file.path(out_dir, "polychor_median_triangle_prsign.pdf")

p_tri <- plot_corr_triangle_from_summary(sum_dt = sum_dt, vars = ord_vars, out_file = out_file)
print(p_tri)
