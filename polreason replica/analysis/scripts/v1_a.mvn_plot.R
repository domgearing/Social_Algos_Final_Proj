################################################################################
# BVN HDR ellipses: uncertainty in pairwise correlations across raters
#
# Overview
# --------
# This module visualizes uncertainty in pairwise belief–belief correlations by
# converting bootstrap draws of correlation coefficients into bivariate normal
# (BVN) highest-density region (HDR) ellipses.
#
# For each rater (GSS and one or more LLMs), we assume latent variables
# (X1, X2) ~ N(0, Sigma) with unit variances and correlation rho. For a chosen
# probability mass `prob` (e.g., 0.50), the HDR boundary is the ellipse defined by:
#     x' Sigma^{-1} x = qchisq(prob, df = 2)
#
# Workflow
# --------
#   1) Load per-rater bootstrap correlation matrices from disk.
#   2) For a specified belief pair (belief_x, belief_y), extract rho from each
#      bootstrap draw (NA if missing).
#   3) For each rho draw, build Sigma = [[1, rho], [rho, 1]] and compute the HDR
#      ellipse boundary points (ellipse_points()).
#   4) Plot one ellipse per bootstrap draw (optional) and overlay the median
#      ellipse per set (optional), with a legend reporting median and 95% interval.
#   5) Optionally iterate over all LLM-eligible belief pairs and save one PDF per
#      pair, using filename prefixes based on correlation magnitude for easy sorting.
#
# Key design choices
# ------------------
#   - Chi-square cutoff for 2D HDRs:
#       The ellipse corresponds to a constant Mahalanobis radius set by
#       qchisq(prob, df = 2).
#   - Numerical stability:
#       Sigma square-roots are computed via eigen-decomposition; small negative
#       eigenvalues (from numerical error) are truncated to zero.
#   - LLM eligibility for batch plotting:
#       When plotting all pairs, we only plot belief pairs that co-occur in at
#       least one LLM bootstrap matrix, to avoid generating figures for pairs
#       that cannot be extracted for any non-GSS rater.
#
# Prerequisites (provided elsewhere in the project)
# ------------------------------------------------
# This module assumes the following objects/functions exist in the workspace:
#   - BASE_OUT_DIR : directory containing "<rater>-<YEAR>" subdirectories
#   - BASE_VIZ_DIR : directory to write output PDFs
#   - available_raters(base_out_dir, year, exclude=...) -> character vector
#   - load_corr_for_rater(rater, base_out_dir, year, suffix?, strict?) -> list
#
# Output
# ------
#   - plot_bvn_belief_pair(): draws to the current device.
#   - plot_all_bvn_pairs(): writes one PDF per eligible belief pair under:
#       file.path(BASE_VIZ_DIR, paste0("mvn_", YEAR))/
################################################################################



#' Compute boundary points for a BVN highest-density ellipse
#'
#' Computes boundary coordinates for the highest-density (probability) region of
#' a bivariate normal distribution with mean \eqn{(0, 0)} and covariance matrix
#' \code{Sigma}. The boundary is the ellipse given by the chi-square cutoff with
#' \eqn{df = 2}.
#'
#' @param Sigma A \eqn{2 \times 2} symmetric, positive semi-definite covariance
#'   matrix.
#' @param prob Probability mass contained within the ellipse (e.g. \code{0.5} for
#'   a 50% HDR region). Must be in \code{(0, 1)}.
#' @param n Integer number of boundary points to return (controls smoothness).
#'
#' @return An \eqn{n \times 2} numeric matrix with columns \code{"x"} and \code{"y"}
#'   giving ellipse boundary coordinates.
#'
#' @details
#' The ellipse boundary corresponds to the quadratic form
#' \deqn{x^\top \Sigma^{-1} x = q_{\chi^2_2}(\mathrm{prob}).}
#' A symmetric square-root of \code{Sigma} is computed via an eigen-decomposition.
#' Any small negative eigenvalues (from numerical issues) are truncated to zero.
#'
#' @examples
#' Sigma <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
#' xy <- ellipse_points(Sigma, prob = 0.5, n = 200)
#' plot(xy, type = "l", asp = 1)
#'
#' @keywords internal
ellipse_points <- function(Sigma, prob = 0.5, n = 361L) {
  r     <- sqrt(qchisq(prob, df = 2))
  theta <- seq(0, 2 * base::pi, length.out = n)
  U     <- rbind(cos(theta), sin(theta))
  
  e <- eigen(Sigma, symmetric = TRUE)
  A <- e$vectors %*% diag(sqrt(pmax(e$values, 0))) %*% t(e$vectors)
  
  XY <- t(A %*% (r * U))
  colnames(XY) <- c("x", "y")
  XY
}



#' Construct a bivariate correlation matrix from a correlation coefficient
#'
#' Builds the \eqn{2 \times 2} correlation matrix for a bivariate normal model
#' from a scalar correlation \code{rho}.
#'
#' @param rho Numeric scalar correlation. Values are clamped to
#'   \code{[-0.9999, 0.9999]} to avoid degeneracy in downstream computations.
#'
#' @return A \eqn{2 \times 2} numeric matrix
#'   \code{matrix(c(1, rho, rho, 1), 2, 2)}.
#'
#' @examples
#' bvn_sigma(0.3)
#'
#' @keywords internal
bvn_sigma <- function(rho) {
  rho <- pmax(pmin(rho, 0.9999), -0.9999)
  matrix(c(1, rho, rho, 1), 2, 2)
}



#' Extract bootstrap correlations for a single belief pair
#'
#' Given a list of bootstrap correlation matrices (as returned by
#' \code{\link{load_corr_for_rater}}), extracts the correlation
#' \eqn{\rho(X, Y)} for a specified belief pair from each draw.
#'
#' @param corr_list List of bootstrap correlation matrices (and/or \code{NULL}
#'   entries). Each non-NULL element should be a square numeric matrix with
#'   column names (and row names) including \code{belief_x} and \code{belief_y}.
#' @param belief_x,belief_y Character scalars giving the two belief/variable
#'   names to extract.
#'
#' @return A numeric vector of length \code{length(corr_list)} containing
#'   \code{S[belief_x, belief_y]} for each draw. If an entry is \code{NULL} or the
#'   beliefs are missing from its dimnames, the corresponding output value is
#'   \code{NA_real_}.
#'
#' @details
#' This is intended to produce the per-set vectors used by
#' \code{\link{plot_bvn_sets}} (often after filtering to finite values).
#'
#' @examples
#' \dontrun{
#' corr_list <- load_corr_for_rater("gss", base_out_dir = BASE_OUT_DIR, year = 2024)
#' rhos <- extract_rhos_for_pair(corr_list, "belief_a", "belief_b")
#' summary(rhos)
#' }
#'
#' @keywords internal
extract_rhos_for_pair <- function(corr_list, belief_x, belief_y) {
  vapply(corr_list, function(S) {
    if (is.null(S)) return(NA_real_)
    nm <- colnames(S)
    if (!all(c(belief_x, belief_y) %in% nm)) return(NA_real_)
    S[belief_x, belief_y]
  }, numeric(1L))
}



#' Plot BVN HDR ellipses for multiple sets of correlation draws
#'
#' Given multiple sets of correlation draws \code{rho_sets} (e.g. bootstrap draws
#' per rater/model), plots the corresponding bivariate normal highest-density
#' region (HDR) ellipses and optionally highlights the median ellipse per set.
#'
#' @param rho_sets A list where each element is a numeric vector of correlation
#'   draws (\code{rho}) for one set.
#' @param prob Probability mass for the HDR ellipse (e.g. \code{0.5} gives a 50%
#'   ellipse). Must be in \code{(0, 1)}.
#' @param alpha_fill Transparency for bootstrap ellipses. Either a scalar applied
#'   to all sets or a numeric vector of length \code{length(rho_sets)}.
#' @param cols Optional vector of colors (length = number of sets) used for
#'   bootstrap ellipses and legend entries. If \code{NULL}, distinct hues are
#'   generated. If a set is labeled \code{"GSS"}, its color is forced to black.
#' @param cols_median Optional vector of colors for median ellipses (length =
#'   number of sets). Defaults to \code{cols}. If a set is labeled \code{"GSS"},
#'   its median color is forced to black.
#' @param set_labels Optional character vector of set names used in the legend
#'   (length = number of sets).
#' @param lwd_lines Line width for bootstrap ellipses.
#' @param lwd_median Line width for median ellipses.
#' @param n Integer number of boundary points per ellipse (smoothness).
#' @param add Logical; if \code{TRUE}, draw onto the current plot. If \code{FALSE}
#'   (default), start a new plot.
#' @param xlab,ylab Axis labels passed to \code{plot()} when \code{add = FALSE}.
#' @param main Plot title; if \code{NULL}, a default title is used.
#' @param draw_axes Logical; if \code{TRUE}, draws pretty axes and dashed
#'   reference lines at \code{x=0} and \code{y=0}.
#' @param legend_pos Legend position when \code{legend_outer = FALSE}; passed to
#'   \code{legend()}.
#' @param legend_cex Legend text size.
#' @param fix_limits Logical; if \code{TRUE} and \code{xlim}/\code{ylim} are
#'   \code{NULL}, use consistent symmetric limits for easier comparison across
#'   plots.
#' @param xlim,ylim Optional axis limits. If \code{NULL}, limits are computed
#'   from data (or fixed symmetrically when \code{fix_limits = TRUE}).
#' @param rho_bound Correlation bound used to compute conservative symmetric
#'   limits when \code{fix_limits = TRUE}.
#' @param legend_outer Logical; if \code{TRUE}, draws the legend just outside the
#'   plotting region to the right (requires sufficient right margin).
#' @param legend_pt_lwd Line width for legend points.
#' @param legend_outer_offset Horizontal offset (as a fraction of x-range) used
#'   to place the outside legend.
#' @param plot_bootstraps Logical; if \code{TRUE}, plot one ellipse per rho draw.
#' @param plot_median Logical; if \code{TRUE}, plot the median ellipse per set.
#' @param legend_pch Optional point symbols for legend entries (length = number
#'   of sets). Only used when \code{legend_show_rank_in_pch = FALSE}.
#' @param legend_order_by_abs_rho Logical; if \code{TRUE}, order legend entries
#'   by increasing \code{|median rho|}.
#' @param legend_show_rank_in_pch Logical; if \code{TRUE}, draws rank numbers
#'   inside the legend points (ranked by increasing \code{|median rho|}).
#' @param bold_gss_in_legend Logical; if \code{TRUE} and a label equals \code{"GSS"},
#'   that legend entry is bolded.
#'
#' @return Invisibly, a list with:
#' \itemize{
#'   \item \code{prob}: ellipse probability mass used.
#'   \item \code{colors}: colors used per set (after filtering).
#'   \item \code{labels}: legend labels (after filtering).
#'   \item \code{median_rhos}: median correlation per set.
#'   \item \code{ci_lower}, \code{ci_upper}: 2.5% and 97.5% quantiles per set.
#'   \item \code{xlim}, \code{ylim}: plot limits used.
#' }
#'
#' @details
#' Non-finite values are dropped within each rho set. Sets with \code{NA} median
#' correlation are removed before plotting and from the legend.
#'
#' If \code{legend_outer = TRUE}, the legend is drawn outside the plot region via
#' \code{par(xpd = NA)}; you will typically want extra right margin, e.g.
#' \code{par(mar = par("mar") + c(0, 0, 0, 26))}.
#'
#' @examples
#' set.seed(1)
#' rho_sets <- list(
#'   GSS = rnorm(200, 0.2, 0.05),
#'   LLM = rnorm(200, 0.5, 0.08)
#' )
#' plot_bvn_sets(rho_sets, set_labels = names(rho_sets), prob = 0.5)
#'
#' @keywords internal
plot_bvn_sets <- function(
    rho_sets,
    prob        = 0.5,
    alpha_fill  = 0.35,
    cols        = NULL,
    cols_median = NULL,
    set_labels  = NULL,
    lwd_lines   = 2,
    lwd_median  = 3,
    n           = 361L,
    add         = FALSE,
    xlab        = expression(Latent~X[1]),
    ylab        = expression(Latent~X[2]),
    main        = NULL,
    draw_axes   = TRUE,
    legend_pos  = "topright",
    legend_cex  = 0.9,
    fix_limits  = TRUE,
    xlim        = NULL,
    ylim        = NULL,
    rho_bound   = 0.9999,
    legend_outer         = TRUE,
    legend_pt_lwd        = 1,
    legend_outer_offset  = 0.03,
    plot_bootstraps      = TRUE,
    plot_median          = TRUE,
    legend_pch              = NULL,
    legend_order_by_abs_rho = TRUE,
    legend_show_rank_in_pch = TRUE,
    bold_gss_in_legend      = TRUE
) {
  
  # ---- sanity checks ----
  k <- length(rho_sets)
  
  if (!is.null(set_labels)) {
    stopifnot(length(set_labels) == k)
  }
  
  if (is.null(cols)) {
    cols <- hsv(seq(0, 0.9, length.out = k), s = 0.7, v = 0.85)
  }
  
  is_gss <- rep(FALSE, k)
  if (!is.null(set_labels)) {
    is_gss <- tolower(set_labels) == "gss"
  }
  
  cols[is_gss] <- "black"
  
  if (is.null(cols_median)) {
    cols_median <- cols
  } else {
    stopifnot(length(cols_median) == k)
    cols_median[is_gss] <- "black"
  }
  
  if (length(alpha_fill) == 1L) {
    alpha_fill <- rep(alpha_fill, k)
  } else {
    stopifnot(length(alpha_fill) == k)
  }
  
  rho_clean <- lapply(rho_sets, function(rhos) rhos[is.finite(rhos)])
  
  median_rhos <- vapply(rho_clean, function(rhos) {
    if (!length(rhos)) NA_real_ else median(rhos)
  }, numeric(1L))
  
  ci_lower <- vapply(rho_clean, function(rhos) {
    if (!length(rhos)) NA_real_ else as.numeric(quantile(rhos, 0.025, na.rm = TRUE))
  }, numeric(1L))
  
  ci_upper <- vapply(rho_clean, function(rhos) {
    if (!length(rhos)) NA_real_ else as.numeric(quantile(rhos, 0.975, na.rm = TRUE))
  }, numeric(1L))
  
  keep <- !is.na(median_rhos)
  if (!any(keep)) {
    stop("All median correlations are NA; nothing to plot.")
  }
  
  rho_clean   <- rho_clean[keep]
  median_rhos <- median_rhos[keep]
  ci_lower    <- ci_lower[keep]
  ci_upper    <- ci_upper[keep]
  cols        <- cols[keep]
  cols_median <- cols_median[keep]
  alpha_fill  <- alpha_fill[keep]
  is_gss      <- is_gss[keep]
  if (!is.null(set_labels)) {
    set_labels <- set_labels[keep]
  }
  
  k <- length(rho_clean)
  
  build_set <- function(rhos) {
    if (!length(rhos)) {
      return(list(nonmed = list(), med_r = NA_real_, med_el = NULL))
    }
    nonmed <- lapply(rhos, function(r) ellipse_points(bvn_sigma(r), prob, n))
    med_r  <- median(rhos, na.rm = TRUE)
    med_el <- ellipse_points(bvn_sigma(med_r), prob, n)
    list(nonmed = nonmed, med_r = med_r, med_el = med_el)
  }
  set_ellipses <- lapply(rho_clean, build_set)
  
  if (fix_limits && is.null(xlim) && is.null(ylim)) {
    r <- sqrt(qchisq(prob, df = 2))
    L <- r * sqrt(1 + abs(rho_bound))
    xlim <- c(-L, L)
    ylim <- c(-L, L)
  }
  
  if (is.null(xlim) || is.null(ylim)) {
    xr <- range(unlist(lapply(seq_len(k), function(s) {
      se <- set_ellipses[[s]]
      xs <- unlist(lapply(se$nonmed, function(m) range(m[, 1])))
      if (!is.null(se$med_el)) xs <- c(xs, range(se$med_el[, 1]))
      xs
    })), na.rm = TRUE)
    
    yr <- range(unlist(lapply(seq_len(k), function(s) {
      se <- set_ellipses[[s]]
      ys <- unlist(lapply(se$nonmed, function(m) range(m[, 2])))
      if (!is.null(se$med_el)) ys <- c(ys, range(se$med_el[, 2]))
      ys
    })), na.rm = TRUE)
    
    pad <- 0.06
    if (is.null(xlim)) xlim <- xr + c(-1, 1) * diff(xr) * pad
    if (is.null(ylim)) ylim <- yr + c(-1, 1) * diff(yr) * pad
  }
  
  if (!add) {
    plot(NA, xlim = xlim, ylim = ylim, asp = 1,
         xlab = xlab, ylab = ylab,
         xaxt = "n", yaxt = "n",
         main = if (is.null(main)) {
           bquote(.(100 * prob) * "% HDR ellipses across sets")
         } else {
           main
         })
    
    if (draw_axes) {
      ticks <- pretty(range(c(xlim, ylim)))
      axis(1, at = ticks)
      axis(2, at = ticks)
      abline(h = 0, v = 0, lty = 3, col = "grey75")
    }
  }
  
  for (s in seq_len(k)) {
    base_col  <- cols[s]
    col_trans <- adjustcolor(base_col, alpha.f = alpha_fill[s])
    
    med_alpha <- if (is_gss[s]) 1 else alpha_fill[s]
    med_alpha <- max(0, min(1, med_alpha))
    med_col   <- adjustcolor(cols_median[s], alpha.f = med_alpha)
    
    se <- set_ellipses[[s]]
    
    if (plot_bootstraps && length(se$nonmed)) {
      for (el in se$nonmed) {
        lines(el[, 1], el[, 2], col = col_trans, lwd = lwd_lines)
      }
    }
    
    if (plot_median && !is.null(se$med_el)) {
      lines(se$med_el[, 1], se$med_el[, 2],
            col = med_col, lwd = lwd_median)
    }
  }
  
  if (!is.null(set_labels)) {
    labels_with_stats <- mapply(function(lbl, med, lo, hi) {
      if (is.na(med)) {
        sprintf("%s (rho = NA)", lbl)
      } else if (is.na(lo) || is.na(hi)) {
        sprintf("%s (rho = %.2f)", lbl, med)
      } else {
        sprintf("%s (rho = %.2f [%.2f, %.2f])", lbl, med, lo, hi)
      }
    }, set_labels, median_rhos, ci_lower, ci_upper, USE.NAMES = FALSE)
    
    ord_strength <- order(abs(median_rhos), na.last = TRUE)
    ord <- if (legend_order_by_abs_rho) ord_strength else seq_len(k)
    
    rank_abs <- integer(k)
    rank_abs[ord_strength] <- seq_along(ord_strength)
    rank_labels <- as.character(rank_abs[ord])
    
    text_font_base <- rep(1L, k)
    if (bold_gss_in_legend && any(is_gss)) {
      text_font_base[is_gss] <- 2L
    }
    text_font_ord <- text_font_base[ord]
    
    legend_labels_ord <- labels_with_stats[ord]
    legend_cols_ord   <- cols[ord]
    
    if (!is.null(legend_pch) && !legend_show_rank_in_pch) {
      stopifnot(length(legend_pch) == k)
      base_pch <- legend_pch[ord]
    } else if (!legend_show_rank_in_pch) {
      base_pch <- rep(16, length(ord))
    } else {
      base_pch <- rep(16, length(ord))
    }
    
    point_cex <- 2.4
    
    if (legend_outer) {
      old_par <- par(no.readonly = TRUE)
      on.exit(par(old_par), add = TRUE)
      par(xpd = NA)
      
      usr <- par("usr")
      xr <- usr[1:2]; yr <- usr[3:4]
      x_off    <- legend_outer_offset * diff(xr)
      x_legend <- xr[2] + x_off
      y_legend <- mean(yr)
      
      lg <- legend(
        x = x_legend, y = y_legend,
        legend = legend_labels_ord,
        pch    = base_pch,
        col    = legend_cols_ord,
        pt.cex = point_cex,
        bty    = "n",
        cex    = legend_cex,
        x.intersp = 0.75,
        y.intersp = 1.25,
        pt.lwd = legend_pt_lwd,
        xjust  = 0, yjust = 0.5,
        text.font = text_font_ord
      )
    } else {
      lg <- legend(
        legend_pos,
        legend = legend_labels_ord,
        pch    = base_pch,
        col    = legend_cols_ord,
        pt.cex = point_cex,
        bty    = "n",
        cex    = legend_cex,
        x.intersp = 0.75,
        y.intersp = 1.25,
        pt.lwd = legend_pt_lwd,
        text.font = text_font_ord
      )
    }
    
    if (legend_show_rank_in_pch) {
      if (!is.null(lg$points)) {
        xs <- lg$points$x
        ys <- lg$points$y
      } else if (!is.null(lg$text)) {
        xs <- lg$text$x - strwidth("  ", cex = legend_cex)
        ys <- lg$text$y
      } else {
        xs <- ys <- NULL
      }
      
      if (!is.null(xs) && !is.null(ys)) {
        text(
          x      = xs,
          y      = ys,
          labels = rank_labels,
          cex    = legend_cex * 0.7,
          col    = "white",
          adj    = c(0.5, 0.5)
        )
      }
    }
  }
  
  invisible(list(
    prob        = prob,
    colors      = cols,
    labels      = set_labels,
    median_rhos = median_rhos,
    ci_lower    = ci_lower,
    ci_upper    = ci_upper,
    xlim        = xlim,
    ylim        = ylim
  ))
}



#' Plot BVN HDR ellipses for a single belief-pair across raters
#'
#' Convenience wrapper that loads correlation draws for a specific belief pair
#' across raters on disk (GSS plus optional LLM raters) and calls
#' \code{\link{plot_bvn_sets}}.
#'
#' @param belief_x,belief_y Character belief names (variables) defining the pair.
#' @param year Integer year used to locate inputs on disk.
#' @param base_out_dir Base directory containing saved correlation objects.
#' @param prob Probability mass for the HDR ellipse.
#' @param llms Optional character vector of non-GSS raters to include. If
#'   \code{NULL}, includes all available non-GSS raters.
#' @param legend_outer,legend_pt_lwd Passed to \code{\link{plot_bvn_sets}}.
#' @param plot_bootstraps,plot_median Passed to \code{\link{plot_bvn_sets}} to
#'   control what gets drawn.
#' @param ... Additional arguments forwarded to \code{\link{plot_bvn_sets}}. If
#'   \code{main} is supplied in \code{...}, it overrides the default title.
#'
#' @return Invisibly, the value returned by \code{\link{plot_bvn_sets}}.
#'
#' @details
#' Uses the helper functions \code{available_raters()}, \code{load_corr_for_rater()},
#' and \code{extract_rhos_for_pair()} to construct \code{rho_sets}. If requested
#' \code{llms} are not found on disk, a warning is emitted and missing raters are
#' skipped.
#'
#' @examples
#' \dontrun{
#' plot_bvn_belief_pair("belief_a", "belief_b", year = 2024, prob = 0.5)
#' }
#'
#' @keywords internal
plot_bvn_belief_pair <- function(
    belief_x,
    belief_y,
    year            = get0("YEAR", ifnotfound = 2024L),
    base_out_dir    = BASE_OUT_DIR,
    prob            = 0.5,
    llms            = NULL,
    legend_outer    = TRUE,
    legend_pt_lwd   = 1,
    plot_bootstraps = TRUE,
    plot_median     = TRUE,
    ...
) {
  raters_all <- available_raters(base_out_dir = base_out_dir, year = year)
  
  has_gss <- "gss" %in% raters_all
  non_gss <- setdiff(raters_all, "gss")
  
  if (is.null(llms)) {
    llm_raters <- non_gss
  } else {
    missing_llm <- setdiff(llms, raters_all)
    if (length(missing_llm) > 0L) {
      warning("Some requested LLM raters not found on disk: ",
              paste(missing_llm, collapse = ", "))
    }
    llm_raters <- intersect(llms, non_gss)
  }
  
  raters <- c(if (has_gss) "gss" else character(0L), llm_raters)
  if (!length(raters)) {
    stop("No raters available for this belief pair.")
  }
  
  rho_sets <- lapply(raters, function(r) {
    corr_list <- load_corr_for_rater(r, base_out_dir = base_out_dir, year = year)
    extract_rhos_for_pair(corr_list, belief_x, belief_y)
  })
  
  set_labels <- ifelse(raters == "gss", "GSS", raters)
  
  dots <- list(...)
  if ("main" %in% names(dots)) {
    main_arg <- dots$main
    dots$main <- NULL
  } else {
    main_arg <- sprintf(
      "%s  ×  %s: %g%% HDR BVN ellipses",
      belief_x, belief_y, 100 * prob
    )
  }
  
  base_args <- list(
    rho_sets        = rho_sets,
    prob            = prob,
    set_labels      = set_labels,
    legend_outer    = legend_outer,
    legend_pt_lwd   = legend_pt_lwd,
    plot_bootstraps = plot_bootstraps,
    plot_median     = plot_median,
    main            = main_arg
  )
  
  do.call(plot_bvn_sets, c(base_args, dots))
}



#' Plot and save BVN HDR ellipses for all eligible belief pairs
#'
#' Iterates over belief pairs and saves one PDF per pair by calling
#' \code{\link{plot_bvn_belief_pair}}.
#'
#' Unlike earlier versions that enumerated pairs from GSS alone, this function
#' *enforces LLM eligibility*: a pair \code{(belief_x, belief_y)} is plotted only
#' if there exists at least one non-GSS rater (LLM) whose saved correlation
#' matrices include *both* variables in the same matrix (i.e., the correlation for
#' that pair is in principle extractable for at least one LLM draw).
#'
#' File naming still uses a magnitude prefix for easier sorting: it prefers the
#' absolute median GSS correlation (when available for that pair), and otherwise
#' falls back to the first LLM rater that provides a finite median for the pair.
#'
#' @param year Integer year used to locate inputs on disk.
#' @param base_out_dir Base directory containing saved correlation objects.
#' @param base_viz_dir Base directory to write output figures.
#' @param prob Probability mass for the HDR ellipse.
#' @param llms Optional character vector of non-GSS raters to include. If
#'   \code{NULL}, includes all available non-GSS raters. The same set is also
#'   used to determine which pairs are eligible to be plotted.
#' @param plot_bootstraps,plot_median Passed to \code{\link{plot_bvn_belief_pair}}
#'   (and ultimately \code{\link{plot_bvn_sets}}).
#' @param ... Additional arguments forwarded to \code{\link{plot_bvn_belief_pair}}.
#'
#' @return Invisibly, a list with components:
#' \itemize{
#'   \item \code{beliefs}: character vector of belief names pooled across eligible LLMs.
#'   \item \code{n_pairs}: number of candidate pairs considered.
#'   \item \code{n_plotted}: number of pairs actually plotted (LLM-eligible).
#'   \item \code{out_dir}: directory where PDFs were written.
#'   \item \code{llm_raters}: LLM raters used for eligibility.
#' }
#'
#' @details
#' Output files are written to \code{file.path(base_viz_dir, paste0("mvn_", year))}.
#'
#' When using outside legends (\code{legend_outer = TRUE} in downstream calls),
#' the function increases the right plot margin via
#' \code{par(mar = par("mar") + c(0, 0, 0, 26))}.
#'
#' @examples
#' \dontrun{
#' plot_all_bvn_pairs(
#'   year = 2024,
#'   prob = 0.5,
#'   llms = c("gpt4o", "claude3"),
#'   plot_bootstraps = FALSE,
#'   plot_median = TRUE
#' )
#' }
#'
#' @keywords internal
plot_all_bvn_pairs <- function(
    year            = get0("YEAR", ifnotfound = 2024L),
    base_out_dir    = BASE_OUT_DIR,
    base_viz_dir    = BASE_VIZ_DIR,
    prob            = 0.5,
    llms            = NULL,
    plot_bootstraps = FALSE,
    plot_median     = TRUE,
    ...
) {
  # ---- rater selection (LLMs drive eligibility) ----
  raters_all <- available_raters(base_out_dir = base_out_dir, year = year)
  
  has_gss <- "gss" %in% raters_all
  non_gss <- setdiff(raters_all, "gss")
  
  if (is.null(llms)) {
    llm_raters <- non_gss
  } else {
    missing_llm <- setdiff(llms, raters_all)
    if (length(missing_llm) > 0L) {
      warning("Some requested LLM raters not found on disk: ",
              paste(missing_llm, collapse = ", "))
    }
    llm_raters <- intersect(llms, non_gss)
  }
  
  if (!length(llm_raters)) {
    stop("No LLM raters available; cannot enforce LLM-based pair eligibility.")
  }
  
  # ---- load correlation objects once for eligibility ----
  llm_corrs <- setNames(lapply(llm_raters, function(r) {
    load_corr_for_rater(r, base_out_dir = base_out_dir, year = year)
  }), llm_raters)
  
  beliefs_in_corr_list <- function(corr_list) {
    nm <- unique(unlist(lapply(corr_list, function(S) {
      if (is.null(S)) return(character(0))
      colnames(S)
    })))
    nm[!is.na(nm) & nzchar(nm)]
  }
  
  pair_exists_in_corr_list <- function(corr_list, belief_x, belief_y) {
    any(vapply(corr_list, function(S) {
      if (is.null(S)) return(FALSE)
      nm <- colnames(S)
      isTRUE(all(c(belief_x, belief_y) %in% nm))
    }, logical(1L)))
  }
  
  median_rho_in_corr_list <- function(corr_list, belief_x, belief_y) {
    rhos <- extract_rhos_for_pair(corr_list, belief_x, belief_y)
    rhos <- rhos[is.finite(rhos)]
    if (!length(rhos)) return(NA_real_)
    median(rhos)
  }
  
  # Candidate beliefs: union across LLMs (NOT GSS-driven)
  beliefs <- sort(unique(unlist(lapply(llm_corrs, beliefs_in_corr_list))))
  if (length(beliefs) < 2L) stop("Fewer than 2 beliefs found across LLM raters.")
  
  pairs <- utils::combn(beliefs, 2L, simplify = FALSE)
  
  out_dir <- file.path(base_viz_dir, paste0("mvn_", year))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load GSS once (optional, used only for filename prefix preference)
  corr_gss <- NULL
  if (has_gss) {
    corr_gss <- load_corr_for_rater("gss", base_out_dir = base_out_dir, year = year)
  }
  
  n_plotted <- 0L
  
  for (p in pairs) {
    belief_x <- p[1]
    belief_y <- p[2]
    
    # ---- eligibility enforcement: pair must co-occur in at least one LLM matrix ----
    pair_ok <- any(vapply(names(llm_corrs), function(r) {
      pair_exists_in_corr_list(llm_corrs[[r]], belief_x, belief_y)
    }, logical(1L)))
    if (!pair_ok) next
    
    # ---- filename prefix: prefer GSS median, else first LLM finite median ----
    prefix_med <- NA_real_
    
    if (!is.null(corr_gss) && pair_exists_in_corr_list(corr_gss, belief_x, belief_y)) {
      prefix_med <- median_rho_in_corr_list(corr_gss, belief_x, belief_y)
    }
    
    if (!is.finite(prefix_med)) {
      for (r in names(llm_corrs)) {
        if (pair_exists_in_corr_list(llm_corrs[[r]], belief_x, belief_y)) {
          m <- median_rho_in_corr_list(llm_corrs[[r]], belief_x, belief_y)
          if (is.finite(m)) {
            prefix_med <- m
            break
          }
        }
      }
    }
    
    rho_prefix <- if (is.finite(prefix_med)) sprintf("%.3f", 100 * abs(prefix_med)) else "NA"
    file_name  <- sprintf("%s_%s_%s.pdf", rho_prefix, belief_x, belief_y)
    file_path  <- file.path(out_dir, file_name)
    
    pdf(file_path, width = 11.5, height = 7.5)
    par(mar = par("mar") + c(0, 0, 0, 26))
    
    plot_bvn_belief_pair(
      belief_x        = belief_x,
      belief_y        = belief_y,
      year            = year,
      base_out_dir    = base_out_dir,
      prob            = prob,
      llms            = llms,            # controls which LLMs are plotted (and matches eligibility set)
      legend_pt_lwd   = 1.5,
      plot_bootstraps = plot_bootstraps,
      plot_median     = plot_median,
      ...
    )
    dev.off()
    
    n_plotted <- n_plotted + 1L
  }
  
  invisible(list(
    beliefs    = beliefs,
    n_pairs    = length(pairs),
    n_plotted  = n_plotted,
    out_dir    = out_dir,
    llm_raters = llm_raters
  ))
}



# ---- RUN ------------------------------------------------------------------- #

plot_all_bvn_pairs(
  base_out_dir    = BASE_OUT_DIR,
  base_viz_dir    = BASE_VIZ_DIR,
  prob            = 0.5,
  legend_outer    = TRUE,
  plot_bootstraps = FALSE,
  plot_median     = TRUE
)
