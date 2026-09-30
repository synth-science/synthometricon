source("paths.R")
# Coverage of PsycTests by each item-text source, counted two ways:
#   overlapping: every record a source can cover, whether or not another source
#                already covers it (semanticnet/aligns match sets from
#                data/processed/{semanticnet,aligns}_matches_all.csv, written by
#                the ingest scripts before their "not yet covered" filter)
#   incremental: records a source adds on top of the tiers before it
#                (synthnet > hunt > semanticnet > aligns; what the treemap colours)
# and against two denominators:
#   records: all PsycTests records (n_records)
#   usage:   PsycInfo usage counts, available for the n_usage records that have
#            been cited at least once; the usage-weighted share is over those
# Writes data/processed/coverage_by_source.csv and prints a markdown table.
suppressMessages({ library(dplyr); library(stringr); library(arrow) })

doi_from_path <- function(path) {
  str_c("10.1037/t", str_sub(str_extract(basename(path), "[0-9]{9}"), 5, 9), "-000")
}
dois_in <- function(parquet) {
  read_parquet(parquet) %>% distinct(path) %>% mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
}

synthnet <- open_dataset("data/raw-extractions-exploded.parquet") %>%
  filter(bucket == "scaled") %>% distinct(path) %>% collect() %>%
  mutate(DOI = doi_from_path(path)) %>% pull(DOI) %>% unique()
hunt <- dois_in("data/restricted/scale-hunt-extractions-exploded.parquet")
semnet_fill <- dois_in("data/semanticnet/semanticnet-extractions-exploded.parquet")
aligns_fill <- dois_in("data/aligns/aligns-extractions-exploded.parquet")
if (file.exists("data/processed/fill_tier_dedupe_aliases.csv")) {
  al <- read.csv("data/processed/fill_tier_dedupe_aliases.csv")
  hunt <- union(hunt, al$dropped_DOI[al$dropped_tier == "hunt"])
  semnet_fill <- union(semnet_fill, al$dropped_DOI[al$dropped_tier == "semanticnet"])
  aligns_fill <- union(aligns_fill, al$dropped_DOI[al$dropped_tier == "aligns"])
}
semnet_all <- union(semnet_fill, read.csv("data/processed/semanticnet_matches_all.csv")$DOI)
aligns_all <- union(aligns_fill, read.csv("data/processed/aligns_matches_all.csv")$DOI)

records <- readRDS(PSYC_RECORDS) %>% distinct(DOI)
usage <- readRDS(PSYC_INFO) %>% group_by(DOI) %>%
  summarise(usage_count = sum(usage_count, na.rm = TRUE), .groups = "drop")
n_records <- nrow(records)
n_usage <- nrow(usage)
total_usage <- sum(usage$usage_count)

sets_overlap <- list(synthnet = synthnet, hunt = hunt, semanticnet = semnet_all, aligns = aligns_all)
sets_incr <- list(
  synthnet = synthnet,
  hunt = setdiff(hunt, synthnet),
  semanticnet = setdiff(semnet_fill, union(synthnet, hunt)),
  aligns = setdiff(aligns_fill, Reduce(union, list(synthnet, hunt, semnet_fill)))
)
sets_overlap$any_source <- Reduce(union, sets_overlap)
sets_incr$any_source <- Reduce(union, sets_incr)

summarise_set <- function(dois) {
  dois <- intersect(dois, records$DOI)
  tibble(records_n = length(dois),
         records_share = length(dois) / n_records,
         usage_n = sum(usage$DOI %in% dois),
         usage_share = sum(usage$usage_count[usage$DOI %in% dois]) / total_usage)
}
tab <- bind_rows(
  bind_rows(lapply(sets_overlap, summarise_set), .id = "source") %>% mutate(counting = "overlapping"),
  bind_rows(lapply(sets_incr, summarise_set), .id = "source") %>% mutate(counting = "incremental")
) %>%
  mutate(n_records = n_records, n_usage = n_usage) %>%
  select(counting, source, records_n, records_share, usage_n, usage_share, n_records, n_usage)
write.csv(tab, "data/processed/coverage_by_source.csv", row.names = FALSE)

cat(sprintf("PsycTests records: %d; records with PsycInfo usage data: %d (total usage %d)\n\n",
            n_records, n_usage, total_usage))
cat("| counting | source | records | share of all records | records with usage data | usage share |\n|---|---|---|---|---|---|\n")
for (i in seq_len(nrow(tab))) {
  cat(sprintf("| %s | %s | %s | %.1f %% | %s | %.1f %% |\n", tab$counting[i], tab$source[i],
              format(tab$records_n[i], big.mark = ","), 100 * tab$records_share[i],
              format(tab$usage_n[i], big.mark = ","), 100 * tab$usage_share[i]))
}
