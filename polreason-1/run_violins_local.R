# run_violins_local.R
# This script runs only the constraint violin visualizations using local data.

# 1. Clear environment
rm(list = ls())
options(scipen = 999)

# 2. Set Year
YEAR <- 2024

# 3. Source Config and Utilities
cat("Loading configuration and utilities...\n")
source("analysis/scripts/0.config.R")
source("analysis/scripts/v.common_utils.R")

# 4. Define the specific rater list
RATERrs_list <- c(
  "gss", 
  "mistralai_mistral-nemo",      # Standard Nemo
  "mistralai_mistral-nemo_OLD", 
  "Nemo_2", 
  "Nemo_v3", 
  "nemo_4", 
  "nemo_5", 
  "dom nemo"                     # Space version
)

cat("Generating constraint violins for:", paste(RATERrs_list, collapse = ", "), "\n")

# 5. Run the constraint statistics and violin plotting script
# This script reads the _v2.rds files and saves PDFs to analysis/viz/
source("analysis/scripts/v2_a.constraint_stats.R")

cat("\nDone! Check 'analysis/viz/constraint_violins_2024/' for your plots.\n")
