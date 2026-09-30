source("paths.R")
# Revised coverage treemap: SynthNet (scaled) + scale hunt + SemanticNet fills.
# Mirrors the treemap logic of 5_coverage_synthnet.Rmd with a source-tier coloring.
suppressMessages({
  library(dplyr); library(stringr); library(arrow); library(plotly)
})

doi_from_path <- function(path) {
  str_c("10.1037/t", str_sub(str_extract(basename(path), "[0-9]{9}"), 5, 9), "-000")
}

synthnet_dois <- open_dataset("data/raw-extractions-exploded.parquet") %>%
  filter(bucket == "scaled") %>% distinct(path) %>% collect() %>%
  mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
hunt_dois <- read_parquet("data/restricted/scale-hunt-extractions-exploded.parquet") %>%
  distinct(path) %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
semnet_dois <- read_parquet("data/semanticnet/semanticnet-extractions-exploded.parquet") %>%
  distinct(path) %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
aligns_dois <- read_parquet("data/aligns/aligns-extractions-exploded.parquet") %>%
  distinct(path) %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()

# content-deduplicated records (dedupe_fill_tiers.R): their items live in the
# corpus under a near-identical sibling record, so the DOI still counts as
# covered, in the tier that originally retrieved it
if (file.exists("data/processed/fill_tier_dedupe_aliases.csv")) {
  al <- read.csv("data/processed/fill_tier_dedupe_aliases.csv")
  hunt_dois <- union(hunt_dois, al$dropped_DOI[al$dropped_tier == "hunt"])
  semnet_dois <- union(semnet_dois, al$dropped_DOI[al$dropped_tier == "semanticnet"])
  aligns_dois <- union(aligns_dois, al$dropped_DOI[al$dropped_tier == "aligns"])
}

psyctests <- readRDS(PSYC_RECORDS)
psyctests_info <- readRDS(PSYC_INFO)

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
  source = case_when(
    DOI %in% synthnet_dois ~ "synthnet",
    DOI %in% hunt_dois ~ "hunt",
    DOI %in% semnet_dois ~ "semanticnet",
    DOI %in% aligns_dois ~ "aligns",
    TRUE ~ "missing"
  ))

set.seed(42)
tests <- psyctests_info %>%
  group_by(DOI, shortName, Name, source) %>%
  summarise(n = sum(usage_count, na.rm = TRUE), parent = "", .groups = "drop") %>%
  arrange(runif(n()))

source("treemap_functions.R")
pal <- readRDS("data/palette.rds")
pal <- rep(pal, length.out = nrow(tests) + 1)
tier <- c("root", tests$source)
pal <- case_when(
  tier == "synthnet" ~ pal,          # SynthNet keeps the original colorful palette
  tier == "hunt" ~ "#E8590C",        # scale hunt: orange
  tier == "semanticnet" ~ "#1971C2", # SemanticNet: blue
  tier == "aligns" ~ "#2F9E44",      # Larsen/aligns public repositories: green
  tier == "missing" ~ "#EEEEEE",
  TRUE ~ pal                          # root
)

# usage-weighted coverage by tier
cov <- tests %>% group_by(source) %>%
  summarise(tests_n = n(), usage = sum(n)) %>%
  mutate(usage_share = usage / sum(usage), tests_share = tests_n / sum(tests_n))
print(cov)

p <- treemap_graph(tests, colors = pal)

# always save the interactive widget; PNG additionally when kaleido is available
# (i.e. when run from RStudio/knitr with the r-reticulate-test env, as in the Rmd)
htmlwidgets::saveWidget(p, "figures/treemap_coverage_sources.html", selfcontained = TRUE)
tryCatch({
  if (!reticulate::py_available()) {
    reticulate::use_miniconda("r-reticulate-test")
    invisible(reticulate::py_available(initialize = TRUE))
  }
  save_image(p, "figures/treemap_coverage_sources.png", width = 1400, height = 700, scale = 5)
  cat("PNG saved\n")
}, error = function(e) message("PNG skipped (", conditionMessage(e), ") - knit the Rmd to render it"))

# --- pooled coverage: original palette, sources not distinguished -------------
# colourful = covered by any source, grey = missing
covered <- tests$source != "missing"
pal_all <- readRDS("data/palette.rds")
pal_all <- rep(pal_all, length.out = nrow(tests) + 1)
pal_all <- if_else(c(FALSE, covered), pal_all, "#EEE")

cat(sprintf("pooled coverage: %d/%d records (%.1f%%), %.1f%% of usage\n",
            sum(covered), length(covered), 100 * mean(covered),
            100 * sum(tests$n[covered]) / sum(tests$n)))

p_all <- treemap_graph(tests, colors = pal_all)

htmlwidgets::saveWidget(p_all, "figures/treemap_coverage_all.html", selfcontained = TRUE)
tryCatch({
  save_image(p_all, "figures/treemap_coverage_all.png", width = 1400, height = 700, scale = 5)
  cat("pooled PNG saved\n")
}, error = function(e) message("pooled PNG skipped (", conditionMessage(e), ") - knit the Rmd to render it"))
