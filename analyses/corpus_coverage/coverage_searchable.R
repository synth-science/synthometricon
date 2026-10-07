source("paths.R")
# Coverage of what is actually SEARCHABLE: like the pooled treemap in scale_hunt_treemap.R, but a PsycTests
# record only counts as covered if at least one of its items survives the production postprocessing filters
# (assemble/postprocess.py FILTERS). Ability/objective tests, intake forms, non-rating-scale, image-based,
# over-long and empty items, unscaled items and failed extractions are therefore not covered.
# The `sample_version` filter applies when the parquet carries `is_sample_version` (the final extraction does).
#
# The git-ignored inputs (extraction and fill-tier parquets) are read from CORPUS_LOCAL_DATA (default "data").
# Same seed and ordering as scale_hunt_treemap.R, so the tile layout is identical and only the colours change.
suppressMessages({
  library(dplyr); library(stringr); library(arrow); library(plotly)
})

LOCAL <- Sys.getenv("CORPUS_LOCAL_DATA", "data")
ITEM_TEXT_CHARS_MAX <- 250  # assemble/postprocess.py

doi_from_path <- function(path) {
  str_c("10.1037/t", str_sub(str_extract(basename(path), "[0-9]{9}"), 5, 9), "-000")
}

filter_cols <- c("path", "has_errors", "bucket", "item_item_type", "item_item_text", "item_has_image",
                 "is_sample_version", "meta_intake_form", "meta_objective_measure")

# rows that survive postprocessing; `apa` = TRUE for the PsycTests extractions, where a missing
# intake-form judgement also drops the row (as in postprocess._filter_intake_form)
searchable_dois <- function(file, apa) {
  d <- open_dataset(file) %>% select(any_of(filter_cols)) %>% collect()
  if (!"is_sample_version" %in% names(d)) d$is_sample_version <- FALSE
  d %>%
    filter(!coalesce(has_errors, FALSE),
           !coalesce(is_sample_version, FALSE),
           bucket %in% c("scaled", "orphan"),
           is.na(item_item_type) | item_item_type == "rating_scale",
           !is.na(item_item_text), nchar(str_trim(item_item_text)) > 0,
           nchar(str_trim(item_item_text)) <= ITEM_TEXT_CHARS_MAX,
           !coalesce(item_has_image, FALSE),
           if (apa) !coalesce(meta_intake_form, TRUE) else !coalesce(meta_intake_form, FALSE),
           !coalesce(meta_objective_measure, FALSE)) %>%
    distinct(path) %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
}
# as in scale_hunt_treemap.R (before postprocessing)
covered_dois <- function(file, scaled_only = FALSE) {
  d <- open_dataset(file)
  if (scaled_only) d <- d %>% filter(bucket == "scaled")
  d %>% distinct(path) %>% collect() %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
}

files <- list(synthnet = file.path(LOCAL, "raw-extractions-exploded.parquet"),
              hunt = file.path(LOCAL, "restricted", "scale-hunt-extractions-exploded.parquet"),
              semanticnet = file.path(LOCAL, "semanticnet", "semanticnet-extractions-exploded.parquet"),
              aligns = file.path(LOCAL, "aligns", "aligns-extractions-exploded.parquet"))
before <- list(synthnet = covered_dois(files$synthnet, scaled_only = TRUE),
               hunt = covered_dois(files$hunt), semanticnet = covered_dois(files$semanticnet),
               aligns = covered_dois(files$aligns))
after <- list(synthnet = searchable_dois(files$synthnet, apa = TRUE),
              hunt = searchable_dois(files$hunt, apa = FALSE),
              semanticnet = searchable_dois(files$semanticnet, apa = FALSE),
              aligns = searchable_dois(files$aligns, apa = FALSE))

# content-deduplicated records (dedupe_fill_tiers.R) count as covered when their kept sibling is
al <- read.csv("data/processed/fill_tier_dedupe_aliases.csv")
for (tier in c("hunt", "semanticnet", "aligns")) {
  a <- al[al$dropped_tier == tier, ]
  before[[tier]] <- union(before[[tier]], a$dropped_DOI)
  after[[tier]] <- union(after[[tier]], a$dropped_DOI[a$kept_DOI %in% unlist(after)])
}

psyctests <- readRDS(PSYC_RECORDS)
psyctests_info <- readRDS(PSYC_INFO)
tier_of <- function(DOI, cov) case_when(
  DOI %in% cov$synthnet ~ "synthnet", DOI %in% cov$hunt ~ "hunt",
  DOI %in% cov$semanticnet ~ "semanticnet", DOI %in% cov$aligns ~ "aligns", TRUE ~ "missing")

psyctests_info <- psyctests_info %>%
  left_join(psyctests %>% select(DOI, Acronym = first_acronym), by = "DOI") %>%
  mutate(shortName = coalesce(Acronym, Name)) %>%
  mutate(shortName = case_when(
    Name == "trail making test" ~ "TMT",
    Name == "alcohol use disorders identification test" ~ "AUDIT",
    Name == "perceived stress scale" ~ "PSS",
    Name == "beck anxiety inventory" ~ "BAI",
    Name == "positive and negative affect scale" ~ "PANAS-B",
    Name == "center for epidemiological studies depression scale" ~ "CESD",
    Name == "stroop color and word test" ~ "SCWT",
    Name == "clinician-administered ptsd scale" ~ "CAPS",
    shortName == "WHO WMH-CIDI" ~ "WMH-CIDI",
    Name == "barthel index" ~ "ADL",
    TRUE ~ shortName
  ),
  source = tier_of(DOI, before), searchable = tier_of(DOI, after))

set.seed(42)
tests <- psyctests_info %>%
  group_by(DOI, shortName, Name, source, searchable) %>%
  summarise(n = sum(usage_count, na.rm = TRUE), parent = "", .groups = "drop") %>%
  arrange(runif(n()))

# coverage of records (all PsycTests records) and of documented uses, before vs after postprocessing
records <- unique(psyctests$DOI)
usage <- psyctests_info %>% group_by(DOI) %>% summarise(u = sum(usage_count, na.rm = TRUE), .groups = "drop")
share <- function(dois) {
  dois <- intersect(dois, records)
  c(records_n = length(dois), records_share = length(dois) / length(records),
    usage_share = sum(usage$u[usage$DOI %in% dois]) / sum(usage$u))
}
cov <- bind_rows(lapply(c("before", "after"), function(stage) {
  x <- if (stage == "before") before else after
  bind_rows(lapply(names(x), function(t) c(stage = stage, tier = t, share(x[[t]]))),
            c(stage = stage, tier = "any_source", share(Reduce(union, x))))
}))
write.csv(cov, "data/processed/coverage_searchable.csv", row.names = FALSE)
print(cov, n = Inf)
tmt <- tests %>% filter(shortName %in% c("TMT", "SCWT", "COWAT", "BDI", "PHQ-9"))
print(tmt %>% select(shortName, source, searchable, n))

source("treemap_functions.R")
pal <- readRDS("data/palette.rds")
pal <- rep(pal, length.out = nrow(tests) + 1)
covered <- tests$searchable != "missing"
pal <- if_else(c(FALSE, covered), pal, "#EEE")
used <- tests$n > 0  # zero-area tiles: 26 DOIs have rows with only NA usage
cat(sprintf("searchable: %d/%d records with usage (%.1f%%), %.1f%% of usage\n",
            sum(covered & used), sum(used), 100 * mean(covered[used]), 100 * sum(tests$n[covered]) / sum(tests$n)))

p <- treemap_graph(tests, colors = pal)
htmlwidgets::saveWidget(p, "figures/treemap_coverage_searchable.html", selfcontained = TRUE)
tryCatch({
  if (!reticulate::py_available()) {
    reticulate::use_miniconda(RETICULATE_ENV)
    invisible(reticulate::py_available(initialize = TRUE))
  }
  save_image(p, "figures/treemap_coverage_searchable.png", width = 1400, height = 700, scale = 5)
  cat("PNG saved\n")
}, error = function(e) message("PNG skipped (", conditionMessage(e), ")"))
