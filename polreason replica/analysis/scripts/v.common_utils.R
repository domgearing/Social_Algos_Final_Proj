#' List available raters on disk for a given year
#'
#' Discovers rater identifiers by listing first-level subdirectories of
#' `base_out_dir` matching "<RATER>-<YEAR>" and stripping the "-<YEAR>" suffix.
#'
#' @param base_out_dir Base directory containing "<rater>-<year>" directories.
#' @param year Integer year used as the directory suffix.
#' @param exclude Character vector of rater identifiers to omit.
#'
#' @return Character vector of raters.
available_raters <- function(
    base_out_dir = BASE_OUT_DIR,
    year        = get0("YEAR", ifnotfound = 2024L),
    exclude     = c("batch_comparison", "persona_size_test")
) {
  pattern    <- sprintf("-%s$", year)
  dirs       <- list.dirs(base_out_dir, recursive = FALSE, full.names = FALSE)
  rater_dirs <- dirs[grepl(pattern, dirs)]
  rs <- sub(pattern, "", rater_dirs)
  rs <- rs[!rs %in% exclude]
  rs
}










#' Load bootstrap correlation matrices for a given rater/year (optionally by subgroup)
#'
#' Supports files:
#'   - polychor_bootstrap.rds
#'   - polychor_bootstrap_loedu.rds
#'   - polychor_bootstrap_hiedu.rds
#'
#' @param rater Character scalar rater id.
#' @param base_out_dir Base output directory containing "<rater>-<year>" subdirs.
#' @param year Integer year.
#' @param suffix One of "", "loedu", "hiedu".
#' @param align_to_mode_varset Logical; if TRUE and filter_to_mode_varset() exists,
#'   align each list to a modal variable set.
#' @param strict Logical; if TRUE, missing/empty files trigger stop(); if FALSE,
#'   they warn and return NULL.
#'
#' @return List of correlation matrices (and/or NULL entries), or NULL if strict=FALSE
#'   and missing/empty.
load_corr_for_rater <- function(
    rater,
    base_out_dir         = BASE_OUT_DIR,
    year                 = get0("YEAR", ifnotfound = 2024L),
    suffix               = "",
    align_to_mode_varset = TRUE,
    strict               = TRUE
) {
  if (!suffix %in% c("", "loedu", "hiedu")) {
    stop('suffix must be one of "", "loedu", "hiedu"')
  }
  
  dir_rater <- file.path(base_out_dir, sprintf("%s-%s", rater, year))
  
  fname <- if (identical(suffix, "")) {
    "polychor_bootstrap.rds"
  } else {
    sprintf("polychor_bootstrap_%s.rds", suffix)
  }
  
  f <- file.path(dir_rater, fname)
  
  if (!file.exists(f)) {
    msg <- paste0(
      "No ", fname, " for rater ", sQuote(rater),
      " and year ", year, " in ", dir_rater
    )
    if (strict) stop(msg) else { warning(msg); return(NULL) }
  }
  
  corr_list <- readRDS(f)
  if (is.null(corr_list) || !length(corr_list)) {
    msg <- paste0("Empty corr_list in ", f, " for rater ", sQuote(rater))
    if (strict) stop(msg) else { warning(msg); return(NULL) }
  }
  
  if (align_to_mode_varset &&
      exists("filter_to_mode_varset", mode = "function", inherits = TRUE)) {
    corr_list <- filter_to_mode_varset(corr_list)
  }
  
  corr_list
}










#' Restrict a list of correlation matrices to a target variable set
#'
#' Given a list of bootstrap correlation matrices, returns a filtered list where
#' each non-NULL matrix is subset to the intersection of its column names and
#' \code{vars_keep}. Matrices with fewer than 2 retained variables are dropped.
#'
#' This is primarily used to restrict GSS matrices to the union of variables
#' appearing in at least one LLM draw, so that “GSS-only” items do not anchor
#' the constraint diagnostics.
#'
#' @param corr_list List of correlation matrices (and/or \code{NULL} entries).
#' @param vars_keep Character vector of variables to keep.
#'
#' @return A list with:
#' \itemize{
#'   \item \code{corr_list}: filtered list of correlation matrices (NULLs removed)
#'   \item \code{bootstrap_id}: integer vector of original bootstrap indices for
#'     the retained matrices
#' }
#'
#' @keywords internal
restrict_corr_list_to_vars <- function(corr_list, vars_keep) {
  if (is.null(corr_list) || !length(corr_list)) {
    return(list(corr_list = NULL, bootstrap_id = integer(0)))
  }
  if (is.null(vars_keep) || !length(vars_keep)) {
    return(list(corr_list = corr_list, bootstrap_id = seq_along(corr_list)))
  }
  
  boot_ids <- seq_along(corr_list)
  out <- vector("list", length(corr_list))
  
  for (i in boot_ids) {
    R <- corr_list[[i]]
    if (is.null(R)) { out[[i]] <- NULL; next }
    
    R <- as.matrix(R)
    
    rn <- rownames(R); cn <- colnames(R)
    if (is.null(rn) || is.null(cn)) { out[[i]] <- NULL; next }
    
    keep <- intersect(cn, vars_keep)
    if (length(keep) < 2L) { out[[i]] <- NULL; next }
    
    out[[i]] <- R[keep, keep, drop = FALSE]
  }
  
  ok <- !vapply(out, is.null, logical(1L))
  list(
    corr_list    = out[ok],
    bootstrap_id = boot_ids[ok]
  )
}










#' Proportion of variance explained by the first principal component
#'
#' Computes the share of total variance explained by the first eigenvalue of a
#' correlation matrix: \eqn{\lambda_1 / \sum_j \lambda_j}.
#'
#' @param R A square numeric correlation matrix (at least 2x2).
#'
#' @return Numeric scalar in \code{[0, 1]} or \code{NA_real_} if \code{R} is invalid.
#'
#' @keywords internal
pc1_var_explained <- function(R) {
  if (is.null(R)) return(NA_real_)
  
  R <- as.matrix(R)
  if (!is.numeric(R)) return(NA_real_)
  if (nrow(R) < 2L || ncol(R) < 2L) return(NA_real_)
  
  ev <- try(eigen(R, symmetric = TRUE, only.values = TRUE)$values,
            silent = TRUE)
  if (inherits(ev, "try-error")) return(NA_real_)
  
  ev <- Re(ev)
  ev <- ev[is.finite(ev)]
  if (!length(ev)) return(NA_real_)
  
  ev  <- pmax(ev, 0)
  tot <- sum(ev)
  if (!is.finite(tot) || tot <= 0) return(NA_real_)
  
  ev[1L] / tot
}










#' Effective dependence of a correlation matrix (Peña & Rodríguez, 2003)
#'
#' Computes the effective dependence statistic:
#' \deqn{De = 1 - |R|^{1/k}}
#' where \eqn{|R|} is the determinant of the correlation matrix and \eqn{k} is
#' its dimension. Computation is performed stably via eigenvalues (geometric mean).
#'
#' If any eigenvalue is non-positive, \eqn{|R| \le 0} and \code{De} is treated as
#' maximal (\code{1}).
#'
#' @param R A square numeric correlation matrix (at least 2x2).
#'
#' @return Numeric scalar in \code{[0, 1]} or \code{NA_real_} if \code{R} is invalid.
#'
#' @keywords internal
effective_dependence <- function(R) {
  if (is.null(R)) return(NA_real_)
  
  R <- as.matrix(R)
  if (!is.numeric(R)) return(NA_real_)
  
  k <- ncol(R)
  if (k < 2L || nrow(R) != k) return(NA_real_)
  
  ev <- try(eigen(R, symmetric = TRUE, only.values = TRUE)$values,
            silent = TRUE)
  if (inherits(ev, "try-error")) return(NA_real_)
  
  ev <- Re(ev)
  ev <- ev[is.finite(ev)]
  if (!length(ev)) return(NA_real_)
  
  if (any(ev <= 0)) return(1)
  
  gm <- exp(mean(log(ev)))
  gm <- max(min(gm, 1), 0)
  
  De <- 1 - gm
  max(min(De, 1), 0)
}