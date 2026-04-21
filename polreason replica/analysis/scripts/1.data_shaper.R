# # --- 1) Build harmonised dataset with beliefs from LLMs, GSS etc. ------- # #
# We put every dataset , including the GSS, in long format to match the LLM 
# data structure

# Load GSS
gss <- readRDS(file = DIR_GSS)

# Assign persona ID
gss$persona_id <- 1:nrow(gss)

# Define Persona Variables
PERSONA_VARS <- c(
  'partyid',
  'polviews',
  'age',
  'sex',
  'race',
  'educ',
  'income',
  'marital',
  'relig',
  'attend'
)

# Get persona attributes from GSS
persona_attrs <- gss[, c("persona_id", intersect(PERSONA_VARS, names(gss)))]
setDT(persona_attrs)
setnames(persona_attrs, old = setdiff(names(persona_attrs), "persona_id"),
         new = paste0("persona_", setdiff(names(persona_attrs), "persona_id")))

# # Clean Persona Attributes

# religion entirely missing
persona_attrs <- persona_attrs[,!'persona_relig']

# partyid
persona_attrs[persona_partyid == 7, persona_partyid := NA_integer_]

# age
breaks <- c(18,24,34,44,54,64,90)
labs   <- sprintf("[%d,%d)", breaks[-length(breaks)], breaks[-1])
persona_attrs$persona_age <- # make this into 12 categories (max levels for other vars)
  cut(
    persona_attrs$persona_age,
    breaks = breaks,
    include_lowest = TRUE,
    right = FALSE,
    labels = labs,
    ordered_result = TRUE
  )
persona_attrs$persona_age <- as.integer(persona_attrs$persona_age)

# sex
persona_attrs$persona_sexfem <- as.numeric(persona_attrs$persona_sex==2)
persona_attrs <- persona_attrs[,!'persona_sex']  

# race - turn to white v. non-white to make sense of order
persona_attrs$persona_rcwhite <- as.numeric(persona_attrs$persona_race==1)
persona_attrs <- persona_attrs[,!'persona_race']

# marital status need to be dichotomised. Married v. non-married, for whatever reason, makes most sense here
persona_attrs$persona_married <- as.numeric(persona_attrs$persona_marital == 1)
persona_attrs <- persona_attrs[,!'persona_marital']

# income - 
# persona_income codes:
# 1-8   : < $10k
# 9-10  : $10k–$14,999
# 11-14 : $15k–$24,999
# 15-16 : $25k–$34,999
# 17-18 : $35k–$49,999
# 19-20 : $50k–$74,999
# 21-24 : $75k–$149,999
# 25-26 : $150k+

persona_attrs[, persona_income := fcase(
  persona_income %in% 1:8,   1L,
  persona_income %in% 9:10,  2L,
  persona_income %in% 11:14, 3L,
  persona_income %in% 15:16, 4L,
  persona_income %in% 17:18, 5L,
  persona_income %in% 19:20, 6L,
  persona_income %in% 21:24, 7L,
  persona_income %in% 25:26, 8L,
  default = NA_integer_
)]

# Join persona attributes to LLM data by matching persona_id to respondent_id
if (tolower(RATER) == 'gss') {
  # ---- GSS path: melt attitudes to long ----
  att_keep <- names(gss)[!names(gss) %in% c('year','persona_id',PERSONA_VARS)] # 
  att_long <- melt(
    as.data.table(gss)[, c("persona_id", att_keep), with = FALSE],
    id.vars = "persona_id",
    variable.name = "variable",
    value.name   = "answer",
    variable.factor = FALSE
  )
  # add meta columns to match LLM schema
  att_long[, `:=`(
    timestamp = as.POSIXct(NA),
    model = "gss",
    run = 1L,
    prompt_tokens = NA_integer_,
    completion_tokens = NA_integer_,
    total_tokens = NA_integer_,
    error = "",
    raw_response = as.character(answer),
    persona_var = FALSE,
    question_short = unname(GSS_QTEXT[variable])
  )]
  
} else {
  # LLM branch: read & standardize
  att_long <- fread(paste0(DIR_LLM,'.csv'))
  
  # ensure required cols exist
  stopifnot("persona_id" %chin% names(att_long), "variable" %chin% names(att_long), "answer" %chin% names(att_long))
  
  # canonical question text where available
  att_long[variable %chin% names(GSS_QTEXT), question_short := unname(GSS_QTEXT[variable])]
  
  # add missing meta columns (esp. persona_var)
  for (nm in c("timestamp","model","run","prompt_tokens","completion_tokens","total_tokens","error","raw_response","persona_var")) {
    if (!(nm %chin% names(att_long))) att_long[, (nm) := NA]
  }
  att_long[is.na(persona_var), persona_var := FALSE]
}

# in both cases, make sure answer is numeric
att_long[, answer := suppressWarnings(as.numeric(answer))]
# merge attitudes with persona attributes
data_j <- merge(att_long, persona_attrs, by = 'persona_id', all.x = TRUE)

# persona rows: must include persona_id in the data passed to melt()
persona_cols <- grep("^persona_", names(persona_attrs)[names(persona_attrs)!='persona_id'], value = TRUE)

pers_long <- melt(
  persona_attrs[persona_id %in% att_long$persona_id][, c("persona_id", persona_cols), with = FALSE],
  id.vars = "persona_id",
  variable.name = "variable",
  value.name = "answer",
  variable.factor = FALSE
)

pers_long[, `:=`(
  persona_var = TRUE,
  question_short = unname(c(
    "persona_partyid"  = "Generally speaking, do you usually think of yourself as a Republican, a Democrat, an Independent, or what?",
    "persona_polviews" = "Where would you place yourself on a liberal–conservative scale?",
    "persona_age"      = "Age of respondent",
    "persona_educ"     = "Highest year of school completed",
    "persona_income"   = "Total family income last year",
    "persona_attend"   = "How often do you attend religious services?",
    "persona_rcwhite"  = "Race: White (1 = yes)",
    "persona_married"  = "Married (1 = yes)",
    "persona_sexfem"   = "Female Sex (1 = yes)"
  )[variable]),
  model = att_long$model[1],
  timestamp = as.POSIXct(NA),
  run = 1L,
  prompt_tokens = NA_integer_,
  completion_tokens = NA_integer_,
  total_tokens = NA_integer_,
  error = "",
  raw_response = as.character(answer)
)]

# columns to keep/bind
keep_cols <- c("persona_id","timestamp","model","variable","question_short","run",
               "answer","prompt_tokens","completion_tokens","total_tokens","error",
               "raw_response","persona_var")

# make sure both data.tables have the same columns
for (nm in keep_cols) {
  if (!(nm %chin% names(data_j)))  data_j[,  (nm) := NA]
  if (!(nm %chin% names(pers_long))) pers_long[, (nm) := NA]
}

# bind and normalize variable names
data_j <- rbindlist(list(
  data_j[, ..keep_cols],
  pers_long[, ..keep_cols]
), use.names = TRUE, fill = TRUE)

data_j[, variable := gsub('^persona_', '', variable)]

# save the harmonised data
saveRDS(object = data_j,file = paste0(DIR_OUT,'/harmonised_data.rds'))
