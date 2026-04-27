################################################################################
# Nemo First-Order Fit Analysis
#
# This script calculates the "First-Order Fit" (1 - Mean TVD) for Nemo models
# by comparing their marginal response distributions to the GSS baseline.
################################################################################

# Load config and utils
YEAR <- 2024
DIR_SCRIPTS <- "analysis/scripts"
source(file.path(DIR_SCRIPTS, "0.config.R"))
source(file.path(DIR_SCRIPTS, "v.common_utils.R"))

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(scales)
})

# 1. Define Nemo models and their clean names
nemo_raters <- c(
  "mistralai_mistral-nemo_OLD",
  "mistralai_mistral-nemo",
  "dom nemo",
  "Dom Nemo v2",
  "Nemo_2",
  "Nemo_v3", 
  "nemo_4",
  "nemo_5"
)

clean_names <- c(
  "mistralai_mistral-nemo_OLD" = "Mistral Nemo (Old)",
  "mistralai_mistral-nemo"     = "Mistral Nemo",
  "dom nemo"                  = "Dom Nemo",
  "Dom Nemo v2"               = "Dom Nemo v2",
  "Nemo_2"                    = "Nemo v2",
  "Nemo_v3"                   = "Nemo v3",
  "nemo_4"                    = "Nemo v4",
  "nemo_5"                    = "Nemo v5"
)

# 2. Load GSS baseline
message("Loading GSS baseline...")
gss_raw <- readRDS(DIR_GSS)
setDT(gss_raw)

# Identify all survey variables (excluding persona and meta vars)
actual_questions <- setdiff(names(gss_raw), c("year", "persona_id", PERSONA_VARS_CANONICAL))

# Compute GSS marginals once
get_marginals <- function(dt, vars) {
  lapply(vars, function(v) {
    counts <- table(dt[[v]])
    if (length(counts) == 0) return(NULL)
    prop.table(counts)
  })
}

gss_marginals <- get_marginals(gss_raw, actual_questions)
names(gss_marginals) <- actual_questions

# 3. Compute TVD for each rater
calc_tvd <- function(p1, p2) {
  # Merge probabilities by response category
  all_cats <- union(names(p1), names(p2))
  p1_full <- setNames(numeric(length(all_cats)), all_cats)
  p2_full <- setNames(numeric(length(all_cats)), all_cats)
  
  p1_full[names(p1)] <- as.numeric(p1)
  p2_full[names(p2)] <- as.numeric(p2)
  
  # TVD = 0.5 * sum|p1 - p2|
  0.5 * sum(abs(p1_full - p2_full))
}

results <- list()

for (rater in nemo_raters) {
  message("Processing rater: ", rater)
  
  # Load harmonized data for this rater (try v2 first, then fallback)
  dir_rater <- file.path(BASE_OUT_DIR, sprintf("%s-%s", rater, YEAR))
  file_h2 <- file.path(dir_rater, paste0("harmonised_data", FILE_SUFFIX, ".rds"))
  file_h1 <- file.path(dir_rater, "harmonised_data.rds")
  
  if (file.exists(file_h2)) {
    file_h <- file_h2
  } else if (file.exists(file_h1)) {
    file_h <- file_h1
  } else {
    warning("Harmonized data not found for ", rater, "; skipping.")
    next
  }
  
  data_r <- readRDS(file_h)
  setDT(data_r)
  
  # Cast to wide to get marginals per variable
  dt_wide_r <- dcast(data_r[variable %in% actual_questions], persona_id + run ~ variable, value.var = "answer")
  
  rater_marginals <- get_marginals(dt_wide_r, actual_questions)
  names(rater_marginals) <- actual_questions
  
  tvds <- numeric()
  for (q in actual_questions) {
    if (!is.null(rater_marginals[[q]]) && !is.null(gss_marginals[[q]])) {
      tvds <- c(tvds, calc_tvd(rater_marginals[[q]], gss_marginals[[q]]))
    }
  }
  
  if (length(tvds) > 0) {
    mean_tvd <- mean(tvds)
    results[[rater]] <- data.table(
      rater = rater,
      clean_name = clean_names[rater],
      mean_tvd = mean_tvd,
      fit_score = 1 - mean_tvd,
      n_vars = length(tvds)
    )
  }
}

if (length(results) == 0) {
  stop("No results computed. Check data availability.")
}

final_dt <- rbindlist(results)
final_dt <- final_dt[order(-fit_score)]
final_dt[, clean_name := factor(clean_name, levels = rev(clean_name))]

# 4. Create premium visualization
message("Creating visualization...")

plot_theme <- theme_minimal(base_size = 14) +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(face = "bold", color = "#2c3e50"),
    plot.title = element_text(face = "bold", size = 18, color = "#2c3e50", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 12, color = "#7f8c8d", margin = margin(b = 20)),
    plot.caption = element_text(size = 10, color = "#95a5a6", margin = margin(t = 20), hjust = 0),
    plot.background = element_rect(fill = "#f8f9fa", color = NA),
    panel.background = element_rect(fill = "#f8f9fa", color = NA)
  )

p <- ggplot(final_dt, aes(x = clean_name, y = fit_score, fill = fit_score)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.3f", fit_score)), 
            hjust = -0.2, fontface = "bold", size = 4.5, color = "#2c3e50") +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.15)),
                     breaks = seq(0, 1, 0.2)) +
  scale_fill_gradientn(colors = c("#e74c3c", "#f39c12", "#2ecc71")) +
  labs(
    title = "First-Order Fit: Nemo Models vs. GSS Baseline",
    subtitle = "Fit score is defined as (1 - Mean TVD) across marginal response distributions.\nA score of 1.0 indicates perfect alignment with human survey responses.",
    x = NULL,
    y = "First-Order Fit Score (Higher is Better)",
    caption = sprintf("Notes: Based on %d survey variables. TVD = Total Variation Distance.\nAnalyzed on %s data.", max(final_dt$n_vars), YEAR)
  ) +
  plot_theme

# Save output
viz_file <- file.path(BASE_VIZ_DIR, "nemo_first_order_fit.pdf")
dir.create(dirname(viz_file), recursive = TRUE, showWarnings = FALSE)

message("Attempting to save plot to: ", viz_file)
message("Current working directory: ", getwd())

# Use standard pdf device directly for better reliability in this environment
pdf(viz_file, width = 10, height = 6)
print(p)
dev.off()

if (file.exists(viz_file)) {
    message("Success! Plot saved to: ", viz_file)
} else {
    message("FAILED to save plot to: ", viz_file)
}

# Also save the raw stats
stats_file <- file.path(BASE_OUT_DIR, "nemo_first_order_stats.csv")
fwrite(final_dt, stats_file)
message("Stats saved to: ", stats_file)
