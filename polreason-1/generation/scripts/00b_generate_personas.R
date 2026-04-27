# Generate Natural Language Personas from GSS 2024 Data
# Converts coded survey responses into readable narrative profiles

library(dplyr)
library(stringr)

# Load the data
gss2024 <- readRDS("../data/gss2024_dellaposta_extract.rds")

# Define natural language mappings for each variable
# Based on GSS codebook

#' Convert EDUC to natural language
educ_to_text <- function(educ) {
  if(is.na(educ)) return(NULL)

  text <- switch(as.character(educ),
    "0" = "I have no formal schooling",
    "1" = "I completed 1st grade",
    "2" = "I completed 2nd grade",
    "3" = "I completed 3rd grade",
    "4" = "I completed 4th grade",
    "5" = "I completed 5th grade",
    "6" = "I completed 6th grade",
    "7" = "I completed 7th grade",
    "8" = "I completed 8th grade",
    "9" = "I completed 9th grade",
    "10" = "I completed 10th grade",
    "11" = "I completed 11th grade",
    "12" = "I completed high school (12th grade)",
    "13" = "I had 1 year of college education",
    "14" = "I had 2 years of college education",
    "15" = "I had 3 years of college education",
    "16" = "I completed 4 years of college (bachelor's degree)",
    "17" = "I had 5 years of college education",
    "18" = "I had 6 years of college education (master's degree)",
    "19" = "I had 7 years of college education",
    "20" = "I had 8 or more years of college education (doctorate or professional degree)",
    NULL
  )
  return(text)
}

#' Convert AGE to natural language
age_to_text <- function(age) {
  if(is.na(age)) return(NULL)
  if(age >= 89) return("I am 89 years or older")
  return(paste0("I am ", age, " years old"))
}

#' Convert SEX to natural language
sex_to_text <- function(sex) {
  if(is.na(sex)) return(NULL)

  text <- switch(as.character(sex),
    "1" = "I am male",
    "2" = "I am female",
    NULL
  )
  return(text)
}

#' Convert RACE to natural language
race_to_text <- function(race) {
  if(is.na(race)) return(NULL)

  text <- switch(as.character(race),
    "1" = "I am White",
    "2" = "I am Black or African American",
    "3" = "I am of another race",
    NULL
  )
  return(text)
}

#' Convert MARITAL to natural language
marital_to_text <- function(marital) {
  if(is.na(marital)) return(NULL)

  text <- switch(as.character(marital),
    "1" = "I am married",
    "2" = "I am widowed",
    "3" = "I am divorced",
    "4" = "I am separated",
    "5" = "I have never been married",
    NULL
  )
  return(text)
}

#' Convert PARTYID to natural language
partyid_to_text <- function(partyid) {
  if(is.na(partyid)) return(NULL)

  text <- switch(as.character(partyid),
    "0" = "I am a strong Democrat",
    "1" = "I am a not very strong Democrat",
    "2" = "I am an independent, close to Democrat",
    "3" = "I am an independent (neither Democrat nor Republican)",
    "4" = "I am an independent, close to Republican",
    "5" = "I am a not very strong Republican",
    "6" = "I am a strong Republican",
    NULL
  )
  return(text)
}

#' Convert POLVIEWS to natural language
polviews_to_text <- function(polviews) {
  if(is.na(polviews)) return(NULL)

  text <- switch(as.character(polviews),
    "1" = "I am extremely liberal",
    "2" = "I am liberal",
    "3" = "I am slightly liberal",
    "4" = "I am moderate (middle of the road)",
    "5" = "I am slightly conservative",
    "6" = "I am conservative",
    "7" = "I am extremely conservative",
    NULL
  )
  return(text)
}

#' Convert RELIG to natural language
relig_to_text <- function(relig) {
  if(is.na(relig)) return(NULL)

  text <- switch(as.character(relig),
    "1" = "My religion is Protestant",
    "2" = "My religion is Catholic",
    "3" = "My religion is Jewish",
    "4" = "I have no religious affiliation",
    "5" = "My religion is Other",
    "6" = "My religion is Buddhism",
    "7" = "My religion is Hinduism",
    "8" = "My religion is Other Eastern",
    "9" = "My religion is Islam",
    "10" = "My religion is Orthodox Christian",
    "11" = "My religion is Christian",
    "12" = "My religion is Native American",
    "13" = "My religion is Inter/Nondenominational",
    NULL
  )
  return(text)
}

#' Convert INCOME to natural language
income_to_text <- function(income) {
  if(is.na(income)) return(NULL)

  text <- switch(as.character(income),
    "1" = "My family income is less than $1,000",
    "2" = "My family income is $1,000 to $2,999",
    "3" = "My family income is $3,000 to $3,999",
    "4" = "My family income is $4,000 to $4,999",
    "5" = "My family income is $5,000 to $5,999",
    "6" = "My family income is $6,000 to $6,999",
    "7" = "My family income is $7,000 to $7,999",
    "8" = "My family income is $8,000 to $9,999",
    "9" = "My family income is $10,000 to $12,499",
    "10" = "My family income is $12,500 to $14,999",
    "11" = "My family income is $15,000 to $17,499",
    "12" = "My family income is $17,500 to $19,999",
    "13" = "My family income is $20,000 to $22,499",
    "14" = "My family income is $22,500 to $24,999",
    "15" = "My family income is $25,000 to $29,999",
    "16" = "My family income is $30,000 to $34,999",
    "17" = "My family income is $35,000 to $39,999",
    "18" = "My family income is $40,000 to $49,999",
    "19" = "My family income is $50,000 to $59,999",
    "20" = "My family income is $60,000 to $74,999",
    "21" = "My family income is $75,000 to $89,999",
    "22" = "My family income is $90,000 to $109,999",
    "23" = "My family income is $110,000 to $129,999",
    "24" = "My family income is $130,000 to $149,999",
    "25" = "My family income is $150,000 to $169,999",
    "26" = "My family income is $170,000 or more",
    NULL
  )
  return(text)
}

#' Convert ATTEND (religious service attendance) to natural language
attend_to_text <- function(attend) {
  if(is.na(attend)) return(NULL)

  text <- switch(as.character(attend),
    "0" = "I never attend religious services",
    "1" = "I attend religious services less than once a year",
    "2" = "I attend religious services about once or twice a year",
    "3" = "I attend religious services several times a year",
    "4" = "I attend religious services about once a month",
    "5" = "I attend religious services 2-3 times a month",
    "6" = "I attend religious services nearly every week",
    "7" = "I attend religious services every week",
    "8" = "I attend religious services more than once a week",
    NULL
  )
  return(text)
}

#' Generate a complete persona for one respondent
generate_persona <- function(respondent_data) {

  statements <- list()

  # Demographics (always include if available)
  if(!is.na(respondent_data$age)) {
    statements <- c(statements, age_to_text(respondent_data$age))
  }

  if(!is.na(respondent_data$sex)) {
    statements <- c(statements, sex_to_text(respondent_data$sex))
  }

  if(!is.na(respondent_data$race)) {
    statements <- c(statements, race_to_text(respondent_data$race))
  }

  if(!is.na(respondent_data$marital)) {
    statements <- c(statements, marital_to_text(respondent_data$marital))
  }

  if(!is.na(respondent_data$educ)) {
    statements <- c(statements, educ_to_text(respondent_data$educ))
  }

  if(!is.na(respondent_data$income)) {
    statements <- c(statements, income_to_text(respondent_data$income))
  }

  # Political views
  if(!is.na(respondent_data$partyid)) {
    statements <- c(statements, partyid_to_text(respondent_data$partyid))
  }

  if(!is.na(respondent_data$polviews)) {
    statements <- c(statements, polviews_to_text(respondent_data$polviews))
  }

  # Religion
  if(!is.na(respondent_data$relig)) {
    statements <- c(statements, relig_to_text(respondent_data$relig))
  }

  if(!is.na(respondent_data$attend)) {
    statements <- c(statements, attend_to_text(respondent_data$attend))
  }

  # Remove NULLs
  statements <- unlist(statements)

  if(length(statements) == 0) {
    return("No profile information available for this respondent.")
  }

  # Combine into natural paragraph
  persona <- paste(statements, collapse=". ")
  persona <- paste0(persona, ".")

  return(persona)
}

#' Generate personas for all respondents
generate_all_personas <- function(data, max_respondents = NULL) {

  if(!is.null(max_respondents)) {
    data <- head(data, max_respondents)
  }

  personas <- data.frame(
    respondent_id = 1:nrow(data),
    persona = character(nrow(data)),
    stringsAsFactors = FALSE
  )

  cat("Generating personas for", nrow(data), "respondents...\n")

  for(i in 1:nrow(data)) {
    if(i %% 100 == 0) cat("  Processed", i, "respondents\n")
    personas$persona[i] <- generate_persona(data[i, ])
  }

  return(personas)
}

# Main execution
cat("GSS 2024 Natural Language Persona Generator\n")
cat("============================================\n\n")

# Generate personas for all respondents
personas <- generate_all_personas(gss2024)

# Save results
saveRDS(personas, "../data/gss2024_personas.rds")
write.csv(personas, "../data/gss2024_personas.csv", row.names = FALSE)

cat("\nPersonas saved to:\n")
cat("  - ../data/gss2024_personas.rds\n")
cat("  - ../data/gss2024_personas.csv\n\n")

# Show examples
cat("Example personas:\n")
cat("=================\n\n")

set.seed(123)
sample_indices <- sample(1:nrow(personas), min(5, nrow(personas)))

for(i in sample_indices) {
  cat("Respondent", personas$respondent_id[i], ":\n")
  cat(strwrap(personas$persona[i], width = 70, prefix = "  "), sep = "\n")
  cat("\n")
}

cat("\nDone! Generated", nrow(personas), "personas.\n")
