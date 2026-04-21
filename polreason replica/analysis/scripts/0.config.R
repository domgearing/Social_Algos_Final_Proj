# # --- Load Packages ------------------------------------------------------ # # 

library(data.table)
library(lavaan)
library(miceRanger)
library(ggplot2)
library(grid)
library(gridExtra)
library(grDevices)
library(scales)
library(ggnewscale)
library(irr)

# # --- helpers ------------------------------------------------------------ # #
# helper: from a wide numeric DT, coerce all ord-vars to ordered factors
to_ordered   <- function(DT, ord_vars) {
  stopifnot(is.data.table(DT))
  
  for (v in ord_vars) {
    vals <- DT[[v]]
    lv <- sort(unique(vals[is.finite(vals)]))
    set(DT, j = v, value = factor(vals, levels = lv, ordered = TRUE))
  }
  
  invisible(DT)
}
to_unordered <- function(DT, ord_vars) {
  stopifnot(is.data.table(DT))
  
  for (v in ord_vars) {
    vals <- DT[[v]]
    if (is.numeric(vals)) {
      lv <- sort(unique(vals[is.finite(vals)]))
    } else {
      lv <- sort(unique(vals[!is.na(vals)]))
    }
    set(DT, j = v, value = factor(vals, levels = lv, ordered = FALSE))
  }
  
  invisible(DT)
}

# Plug this into your existing bootstrap shell:
bootstrap <- function(wide_dt, B = B, seed = NULL, verbose = TRUE) {
  if (!is.null(seed)) set.seed(seed)
  
  # We still bootstrap at the persona (cluster) level
  # but work with row indices instead of merges.
  
  # List of row indices for each persona_id
  idx_by_persona <- split(seq_len(nrow(wide_dt)), wide_dt$persona_id)
  nP <- length(idx_by_persona)
  
  # Preallocate result list
  boots <- vector("list", B)
  
  for (b in seq_len(B)) {
    if (verbose && (b %% max(1L, floor(B / 10))) == 0L) {
      message("Bootstrap ", b, "/", B)
    }
    
    # Two-stage: sample personas, then (within each) sample runs/rows
    sampled_personas_idx <- sample.int(nP, nP, replace = TRUE)
    
    rows <- lapply(sampled_personas_idx, function(i) {
      idx <- idx_by_persona[[i]]
      if (length(idx) == 0L) return(integer(0))
      # same length as original persona cluster, but with replacement
      sample(idx, length(idx), replace = TRUE)
    })
    
    rows <- unlist(rows, use.names = FALSE)
    if (!length(rows)) next
    
    # No merge, just row subset
    boots[[b]] <- wide_dt[rows]
  }
  
  # in case any were skipped (very unlikely)
  boots[!vapply(boots, is.null, logical(1))]
}

# Calculate average from corr or pcorr matrix accounting for names and missing columns 
mean_corr_pairwise <- function(corr_list) {
  # union of all variable names across bootstraps
  all_vars <- sort(unique(unlist(lapply(corr_list, colnames))))
  p <- length(all_vars)
  
  # 3D array to hold each bootstrap mapped to full var set
  arr <- array(NA_real_,
               dim = c(p, p, length(corr_list)),
               dimnames = list(all_vars, all_vars, NULL))
  
  for (b in seq_along(corr_list)) {
    S  <- as.matrix(corr_list[[b]])
    vb <- colnames(S)
    idx <- match(vb, all_vars)
    
    # Put this S into the right rows/cols, leave others as NA
    arr[idx, idx, b] <- S[vb, vb, drop = FALSE]
  }
  
  # Pairwise-complete mean across bootstraps
  S_mean <- apply(arr, c(1, 2), mean, na.rm = TRUE)
  
  # Make sure diagonal is 1
  diag(S_mean) <- 1
  
  # Optionally, drop vars that never appeared with anyone (all NA off-diagonal)
  all_na_offdiag <- apply(S_mean, 1, function(x) all(is.na(x[-which.min(abs(x - 1))])))
  if (any(all_na_offdiag)) {
    keep <- !all_na_offdiag
    S_mean <- S_mean[keep, keep, drop = FALSE]
  }
  
  # Replace any remaining NA's (very rare) with 0 correlation
  S_mean[is.na(S_mean)] <- 0
  
  S_mean
}

# Filters boostraps to ensure varnames are aligned - drops straps unaligned with modal matrix
filter_to_mode_varset <- function(corr_list) {
  # key = sorted set of variable names
  name_key <- function(S) paste(sort(colnames(S)), collapse = "|")
  
  keys <- sapply(corr_list, name_key)
  tab  <- table(keys)
  mode_key <- names(which.max(tab))
  
  keep_idx <- which(keys == mode_key)
  corr_keep <- corr_list[keep_idx]
  
  # Use a reference ordering from the first kept matrix
  ref_vars <- sort(colnames(corr_keep[[1]]))
  
  corr_keep_aligned <- lapply(corr_keep, function(S) {
    S <- as.matrix(S)
    S[ref_vars, ref_vars, drop = FALSE]
  })
  
  message(sprintf(
    "keeping %d bootstraps (dropped %d) with the most frequent variable set.",
    length(corr_keep_aligned), length(corr_list) - length(corr_keep_aligned)
  ))
  
  corr_keep_aligned
}

# # --- Boostrap------------------------------------------------------------ # #
B = 500L

# multiple imputation DGPs
B_MI = 30L

# # --- Define directories ------------------------------------------------- # #
DIR_GSS      <- paste0('generation/data/gss',YEAR,'_dellaposta_extract.rds')

BASE_OUT_DIR <- 'analysis/output'
if (!dir.exists(BASE_OUT_DIR )) dir.create(BASE_OUT_DIR , recursive = TRUE)

BASE_VIZ_DIR  <- 'analysis/viz'
if (!dir.exists(BASE_VIZ_DIR )) dir.create(BASE_VIZ_DIR , recursive = TRUE)


# # --- Define Details on GSS Questions ------------------------------------ # #

GSS_QUESTIONS <- list(
  # Abortion
  abhlth  = list(
    text = "Should it be possible for a pregnant woman to obtain a legal abortion if the woman's own health is seriously endangered by the pregnancy?",
    options = c(`1` = "Yes", `2` = "No")
  ),
  abrape  = list(
    text = "Should it be possible for a pregnant woman to obtain a legal abortion if she became pregnant as a result of rape?",
    options = c(`1` = "Yes", `2` = "No")
  ),
  abpoor  = list(
    text = "Should it be possible for a pregnant woman to obtain a legal abortion if the family has a very low income and cannot afford any more children?",
    options = c(`1` = "Yes", `2` = "No")
  ),
  abdefect = list(
    text = "Should it be possible for a pregnant woman to obtain a legal abortion if there is a strong chance of serious defect in the baby?",
    options = c(`1` = "Yes", `2` = "No")
  ),
  
  # Gun control
  gunlaw = list(
    text = "Would you favor or oppose a law which would require a person to obtain a police permit before he or she could buy a gun?",
    options = c(`1` = "Favor", `2` = "Oppose")
  ),
  
  # Immigration
  immjobs = list(
    text = "Do immigrants take jobs away from people who were born in America, or do immigrants help the American economy by providing cheap labor and new markets?",
    options = c(`1`="Take jobs away (strongly agree)", `2`="2", `3`="3", `4`="Neither", `5`="5", `6`="6", `7`="Help economy (strongly disagree)")
  ),
  immcrime = list(
    text = "Do immigrants increase crime rates in America?",
    options = c(`1`="Very likely", `2`="2", `3`="Neither likely nor unlikely", `4`="4", `5`="Very unlikely")
  ),
  immameco = list(
    text = "Do immigrants improve American society by bringing in new ideas and cultures?",
    options = c(`1`="Agree strongly", `2`="2", `3`="Neither", `4`="4", `5`="Disagree strongly")
  ),
  
  # Sexual morality
  homosex = list(
    text = "What about sexual relations between two adults of the same sex—do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
    options = c(`1`="Always wrong", `2`="Almost always wrong", `3`="Sometimes wrong", `4`="Not wrong at all")
  ),
  premarsx = list(
    text = "If a man and woman have sex relations before marriage, do you think it is always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
    options = c(`1`="Always wrong", `2`="Almost always wrong", `3`="Sometimes wrong", `4`="Not wrong at all")
  ),
  xmarsex = list(
    text = "What is your opinion about a married person having sexual relations with someone other than the marriage partner—is it always wrong, almost always wrong, wrong only sometimes, or not wrong at all?",
    options = c(`1`="Always wrong", `2`="Almost always wrong", `3`="Sometimes wrong", `4`="Not wrong at all")
  ),
  
  # Gender roles
  fechld = list(
    text = "Do you agree or disagree with this statement? A working mother can establish just as warm and secure a relationship with her children as a mother who does not work.",
    options = c(`1`="Strongly agree", `2`="Agree", `3`="Disagree", `4`="Strongly disagree")
  ),
  fefam = list(
    text = "Do you agree or disagree with this statement? It is much better for everyone involved if the man is the achiever outside the home and the woman takes care of the home and family.",
    options = c(`1`="Strongly agree", `2`="Agree", `3`="Disagree", `4`="Strongly disagree")
  ),
  fepresch = list(
    text = "Do you agree or disagree with this statement? A preschool child is likely to suffer if his or her mother works.",
    options = c(`1`="Strongly agree", `2`="Agree", `3`="Disagree", `4`="Strongly disagree")
  ),
  
  # Economic
  eqwlth = list(
    text = "Should government reduce income differences between rich and poor?",
    options = c(`1`="Government should reduce differences", `2`="2", `3`="3", `4`="No government action", `5`="5", `6`="6", `7`="No government action at all")
  ),
  
  # Criminal justice
  cappun = list(
    text = "Do you favor or oppose the death penalty for persons convicted of murder?",
    options = c(`1`="Favor", `2`="Oppose")
  ),
  grass = list(
    text = "Do you think the use of marijuana should be made legal or not?",
    options = c(`1`="Legal", `2`="Not legal")
  ),
  
  # Pornography
  pornlaw = list(
    text = "Which of these statements comes closest to your feelings about pornography laws? 1) There should be laws against the distribution of pornography whatever the age. 2) There should be laws against the distribution of pornography to persons under 18. 3) There should be no laws forbidding the distribution of pornography.",
    options = c(`1`="Illegal to all", `2`="Illegal under 18", `3`="Legal to all")
  ),
  
  # Affirmative action
  affrmact = list(
    text = "Some people say that because of past discrimination, blacks should be given preference in hiring and promotion. Others say that such preference in hiring and promotion of blacks is wrong because it discriminates against whites. What about your opinion—are you for or against preferential hiring and promotion of blacks?",
    options = c(`1`="Favor", `2`="Oppose")
  ),
  
  # Confidence in institutions
  confed = list(
    text = "As far as the people running the federal government are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?",
    options = c(`1`="Great deal", `2`="Only some", `3`="Hardly any")
  ),
  conpress = list(
    text = "As far as the people running the press are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?",
    options = c(`1`="Great deal", `2`="Only some", `3`="Hardly any")
  ),
  consci = list(
    text = "As far as the people running the scientific community are concerned, would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all in them?",
    options = c(`1`="Great deal", `2`="Only some", `3`="Hardly any")
  ),
  
  # Civil liberties
  spkath = list(
    text = "If such a person (an atheist) wanted to make a speech in your community, should he be allowed to speak, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  colath = list(
    text = "Should such a person (an atheist) be allowed to teach in a college or university, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  spkrac = list(
    text = "Consider a person who believes that Blacks are genetically inferior. If such a person wanted to make a speech in your community claiming that Blacks are inferior, should he be allowed to speak, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  colrac = list(
    text = "Should such a person (who believes that Blacks are genetically inferior) be allowed to teach in a college or university, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  spkmslm = list(
    text = "Consider a Muslim clergyman who preaches hatred of the United States. If such a person wanted to make a speech in your community preaching hatred of the United States, should he be allowed to speak, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  colmslm = list(
    text = "Should such a person (a Muslim clergyman who preaches hatred of the United States) be allowed to teach in a college or university, or not?",
    options = c(`1`="Allowed", `2`="Not allowed")
  ),
  
  # Religion
  god = list(
    text = "Which statement comes closest to expressing what you believe about God? 1) I don't believe in God. 2) I don't know whether there is a God and I don't believe there is any way to find out. 3) I don't believe in a personal God, but I do believe in a Higher Power of some kind. 4) I find myself believing in God some of the time, but not at others. 5) While I have doubts, I feel that I do believe in God. 6) I know God really exists and I have no doubts about it.",
    options = c(`1`="Don't believe", `2`="Agnostic", `3`="Higher power", `4`="Believe sometimes", `5`="Believe with doubts", `6`="Know God exists")
  ),
  bible = list(
    text = "Which of these statements comes closest to describing your feelings about the Bible? 1) The Bible is the actual word of God and is to be taken literally, word for word. 2) The Bible is the inspired word of God but not everything in it should be taken literally, word for word. 3) The Bible is an ancient book of fables, legends, history, and moral precepts recorded by men.",
    options = c(`1`="Literal word of God", `2`="Inspired word", `3`="Book of fables")
  )
)

GSS_QTEXT   <- vapply(names(GSS_QUESTIONS), function(v) GSS_QUESTIONS[[v]]$text,   FUN.VALUE = "", USE.NAMES = TRUE)
GSS_OPTIONS <- lapply(GSS_QUESTIONS, `[[`, "options")


# Persona variables as used in the GSS / persona construction.
# We intersect this list with the actual variable names present in the
# correlation matrices, so it is safe to keep variables that may have been
# dropped (e.g., relig).
PERSONA_VARS_CANONICAL <- c(
  "partyid", 
  "polviews",
  "age", 
  "sexfem", 
  "educ", 
  "income",
  "married", 
  "rcwhite"
)
