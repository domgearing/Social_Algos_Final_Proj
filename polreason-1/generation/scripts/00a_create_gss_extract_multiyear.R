#!/usr/bin/env Rscript
# Create GSS Extract for della Posta Analysis (Multi-Year Support)
# This script extracts data for any year for all variables used in della Posta (2020)
# "Pluralistic Collapse: The 'Oil Spill' Model of Mass Opinion Polarization"
#
# Usage:
#   Rscript 00a_create_gss_extract_multiyear.R --year 2024
#   Rscript 00a_create_gss_extract_multiyear.R --year 2016
#   Rscript 00a_create_gss_extract_multiyear.R --year 2008
#   Rscript 00a_create_gss_extract_multiyear.R --year 2000

library(haven)
library(optparse)

# Parse command line arguments
option_list <- list(
  make_option(c("--year"), type="integer", default=2024,
              help="Survey year to extract (2024, 2016, 2008, 2000) [default: %default]")
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)

year <- args$year

# Validate year
if (!(year %in% c(2024, 2016, 2008, 2000))) {
  stop("Error: Year must be one of: 2024, 2016, 2008, 2000")
}

cat("=", rep("=", 68), "\n", sep="")
cat("Creating GSS", year, "Extract for della Posta Analysis\n")
cat("=", rep("=", 68), "\n\n", sep="")

# Load GSS cumulative data (1972-2024)
cat("Loading GSS cumulative data...\n")
gss_full <- read_dta("../data/gss7224_r1.dta")

# Convert to data frame and strip Stata labels
gss_full <- as.data.frame(gss_full)
for(i in 1:ncol(gss_full)) {
  if(inherits(gss_full[[i]], "haven_labelled")) {
    gss_full[[i]] <- as.numeric(gss_full[[i]])
  }
}

# Filter to specified year
gss_year <- gss_full[gss_full$year == year, ]

cat("Total respondents in", year, ":", nrow(gss_year), "\n")
cat("Total variables in full GSS:", ncol(gss_full), "\n\n")

# Define variable categories from della Posta's analysis
# Based on Step1_Variable_Recodes.do line 226

# Core demographic and political variables
core_vars <- c("year", "partyid", "polviews", "age", "sex", "race", "educ",
               "income16", "marital", "relig", "attend")

# Racial attitudes and feeling thermometers
racial_vars <- c("feelblks", "feelasns", "feelhsps", "feelwhts",
                 "racmar", "racdin", "racpush", "racseg", "racopen", "racobjct",
                 "racschol", "racfew", "rachaf", "racmost", "busing", "racpres",
                 "racjob", "racnobuy", "racparty", "racocc", "racinc", "affrmact",
                 "wrkwayup", "blksimp", "helpful", "trust")

# National spending priorities
spending_vars <- c("natspac", "natenvir", "natheal", "natcity", "natcrime",
                   "natdrug", "nateduc", "natrace", "natarms", "nataid", "natfare",
                   "natroad", "natsoc", "natmass", "natpark", "natchld", "natsci", "natenrgy")

# Economic attitudes
economic_vars <- c("eqwlth", "taxrich", "taxmid", "taxpoor", "govcare",
                   "goveqinc", "govedop", "govjobs", "govunemp", "incgap",
                   "workfare", "lessfare")

# Civil liberties and free speech
civil_lib_vars <- c("spkath", "colath", "libath", "spksoc", "colsoc", "libsoc",
                    "spkrac", "colrac", "librac", "spkcom", "colcom", "libcom",
                    "spkmil", "colmil", "libmil", "spkhomo", "colhomo", "libhomo",
                    "spkmslm", "colmslm", "libmslm")

# Criminal justice
crime_vars <- c("cappun", "gunlaw", "courts", "wirtap", "grass", "polhitok")

# Gender and family attitudes
gender_vars <- c("fehome", "fework", "fepres", "fepol", "fechld", "fehelp",
                 "fepresch", "fefam", "era", "febear", "feworkif")

# Abortion attitudes
abortion_vars <- c("abdefect", "abnomore", "abhlth", "abpoor", "abrape",
                   "absingle", "abany")

# Sexual morality
sex_moral_vars <- c("premarsx", "teensex", "xmarsex", "homosex", "pornlaw")

# Religious variables
religion_vars <- c("god", "prayer", "bible", "postlife", "gochurch", "pray")

# Confidence in institutions
confidence_vars <- c("confinan", "conbus", "conclerg", "coneduc", "confed",
                     "conlabor", "conpress", "conmedic", "contv", "conjudge",
                     "consci", "conlegis", "conarmy")

# Immigration attitudes
immigration_vars <- c("letin", "immameco", "immcrime", "immjobs", "immrghts")

# Science and environment
science_vars <- c("scisolve", "scichng", "tempgen", "grncon")

# Combine all variable categories
all_dellaposta_vars <- unique(c(
  core_vars, racial_vars, spending_vars, economic_vars, civil_lib_vars,
  crime_vars, gender_vars, abortion_vars, sex_moral_vars, religion_vars,
  confidence_vars, immigration_vars, science_vars
))

# Keep only variables that exist in the year data
available_vars <- all_dellaposta_vars[all_dellaposta_vars %in% names(gss_year)]
missing_vars <- all_dellaposta_vars[!all_dellaposta_vars %in% names(gss_year)]

cat("Variables requested:", length(all_dellaposta_vars), "\n")
cat("Variables available in", year, ":", length(available_vars), "\n")
cat("Variables not available:", length(missing_vars), "\n\n")

if(length(missing_vars) > 0 && length(missing_vars) <= 30) {
  cat("Missing variables:\n")
  cat("  ", paste(missing_vars, collapse=", "), "\n\n")
}

# Extract the data
gss_year_extract <- gss_year[, available_vars]

# Harmonize income variable name: income16 → income
if ("income16" %in% names(gss_year_extract) && !"income" %in% names(gss_year_extract)) {
  names(gss_year_extract)[names(gss_year_extract) == "income16"] <- "income"
}

# Save as RDS
output_file <- paste0("../data/gss", year, "_dellaposta_extract.rds")
saveRDS(gss_year_extract, file = output_file)

cat("Data saved to:", output_file, "\n")
cat("Dimensions:", nrow(gss_year_extract), "respondents x",
    ncol(gss_year_extract), "variables\n\n")

# Create summary statistics
non_missing <- colSums(!is.na(gss_year_extract))
summary_df <- data.frame(
  variable = names(gss_year_extract),
  n_valid = non_missing,
  pct_valid = round(100 * non_missing / nrow(gss_year_extract), 1)
)
summary_df <- summary_df[order(-summary_df$n_valid), ]

cat("Top 10 variables by response rate:\n")
print(head(summary_df, 10))

cat("\n\nBottom 10 variables by response rate:\n")
print(tail(summary_df, 10))

# Save summary
summary_file <- paste0("../data/gss", year, "_variable_summary.csv")
write.csv(summary_df, summary_file, row.names = FALSE)
cat("\nVariable summary saved to:", summary_file, "\n")

cat("\n")
cat("=", rep("=", 68), "\n", sep="")
cat("GSS", year, "Extract Complete\n")
cat("=", rep("=", 68), "\n", sep="")
