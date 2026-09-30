source("paths.R")
# Ingest the Larsen/aligns corpus (data/aligns/, curated by Kuelpmann & Juenger from
# public repositories: NIH HEAL, Catalogue of Mental Health Measures, Science of
# Behavior Change, Stress Measurement Network, PROMIS) for PsycTests records not yet
# covered by SynthNet (scaled), the scale hunt, or the SemanticNet ingest.
# Gate: distinctive name match (core >= 2 tokens) AND item count within 25% of
# PsycTests expectation (when known). Reverse coding/options unavailable -> NA.
suppressMessages({
  library(dplyr); library(tidyr); library(stringr); library(purrr); library(arrow)
})

norm <- function(x) str_squish(str_replace_all(tolower(x), "[^a-z0-9 ]", " "))
GENERIC <- c("scale","scales","questionnaire","inventory","test","index","survey","measure",
             "checklist","schedule","form","revised","short","brief","version","assessment",
             "interview","rating","the")
core <- function(x) sapply(str_split(norm(x), " "), function(t) paste(setdiff(t, GENERIC), collapse = " "))

d <- read_larsen_instruments() %>%
  # split multi-item cells, clean, dedupe within instrument
  separate_rows(item_text, sep = "\n") %>%
  mutate(item_text = str_squish(item_text)) %>%
  filter(nchar(item_text) >= 3, !str_detect(item_text, "^https?://")) %>%
  distinct(meta_instrument_name, item_text, .keep_all = TRUE)

inst <- d %>% count(meta_instrument_name, name = "n_items") %>%
  mutate(clean = str_squish(str_remove_all(meta_instrument_name, "[(][^)]*[)]")),
         nname = norm(clean), cname = core(clean),
         core_tokens = str_count(cname, "[[:alnum:]]+"))

source("lang_exclude.R")

v <- read.csv("data/semanticnet/psyc_name_variants.csv") %>%
  mutate(nvar = norm(variant), cvar = core(variant))
flags <- read.csv("data/semanticnet/psyc_coverage_flags.csv")
# covered_hunt in the flags file goes stale whenever a hunt batch lands;
# read the hunt parquet directly instead
hunt_dois <- read_parquet("data/restricted/scale-hunt-extractions-exploded.parquet") %>%
  distinct(path) %>%
  mutate(DOI = str_c("10.1037/t", str_sub(str_extract(path, "[0-9]{9}"), 5, 9), "-000")) %>%
  pull(DOI) %>% unique()
semnet_dois <- read_parquet("data/semanticnet/semanticnet-extractions-exploded.parquet") %>%
  distinct(path) %>%
  mutate(DOI = str_c("10.1037/t", str_sub(str_extract(path, "[0-9]{9}"), 5, 9), "-000")) %>%
  pull(DOI) %>% unique()
p <- readRDS(PSYC_RECORDS)
meta <- p %>% transmute(DOI, Name, first_construct, InstrumentType, TestYear, Permissions,
                        expected = number_of_test_items_best_guess)
pi <- readRDS(PSYC_INFO)
usage <- pi %>% group_by(DOI) %>% summarise(usage_count = sum(usage_count, na.rm = TRUE))

cand <- bind_rows(
  inst %>% inner_join(v %>% select(nvar, DOI), by = c("nname" = "nvar"),
                      relationship = "many-to-many") %>% mutate(mtype = "exact"),
  inst %>% filter(core_tokens >= 2, nchar(cname) >= 8) %>%
    inner_join(v %>% filter(nchar(cvar) >= 8) %>% select(cvar, DOI), by = c("cname" = "cvar"),
               relationship = "many-to-many") %>% mutate(mtype = "core")
) %>%
  distinct(meta_instrument_name, n_items, DOI, mtype) %>%
  left_join(flags, by = "DOI") %>% left_join(meta, by = "DOI") %>%
  left_join(usage, by = "DOI") %>%
  mutate(usage_count = coalesce(usage_count, 0),
         covered = covered_synthnet | DOI %in% hunt_dois | DOI %in% semnet_dois,
         count_ok = is.na(expected) | (abs(n_items - expected) / pmax(expected, 1)) <= 0.25) %>%
  # aligns items are English (zero non-ASCII in the corpus): never file them
  # under a translation record
  filter(!is_language_variant(Name), count_ok)

# manual review exclusions: version mis-assignments that pass the count gate
# (12-item GHQ under the GHQ-28 record; adult STAI state form under STAI-for-Children)
BAD_DOI <- c("10.1037/t16058-000", "10.1037/t06497-000")

# full gated match set, regardless of coverage by other tiers (for
# coverage_by_source.R, which reports overlapping coverage per source)
cand %>%
  filter(!DOI %in% BAD_DOI) %>%
  group_by(meta_instrument_name) %>%
  filter(mtype == ifelse(any(mtype == "exact"), "exact", "core")) %>%
  slice_max(usage_count, n = 1, with_ties = FALSE) %>% ungroup() %>%
  group_by(DOI) %>% slice_max(n_items, n = 1, with_ties = FALSE) %>% ungroup() %>%
  transmute(DOI, Name, meta_instrument_name, n_items, usage_count, mtype,
            covered_synthnet, covered_hunt = DOI %in% hunt_dois, covered_semnet = DOI %in% semnet_dois) %>%
  write.csv("data/processed/aligns_matches_all.csv", row.names = FALSE)

cand <- cand %>% filter(!covered)

# exact matches beat core matches; then one best DOI per instrument (highest usage),
# one instrument per DOI (most items)
fills <- cand %>%
  filter(!DOI %in% BAD_DOI) %>%
  group_by(meta_instrument_name) %>%
  filter(mtype == ifelse(any(mtype == "exact"), "exact", "core")) %>%
  slice_max(usage_count, n = 1, with_ties = FALSE) %>% ungroup() %>%
  group_by(DOI) %>% slice_max(n_items, n = 1, with_ties = FALSE) %>% ungroup()

rows <- fills %>%
  mutate(accession = str_c("9999", str_match(DOI, "10[.]1037/t([0-9]{5})-000")[, 2]),
         slug = str_sub(str_replace_all(tolower(Name), "[^a-z0-9]+", "-"), 1, 50)) %>%
  left_join(d, by = "meta_instrument_name", relationship = "many-to-many") %>%
  group_by(DOI) %>% mutate(item_n = row_number()) %>%
  # subscale structure where the corpus provides it (scale_name != total_scale)
  mutate(has_sub = scale_name != "total_scale" & !is.na(scale_name)) %>%
  group_by(DOI, scale_name) %>% mutate(sub_key = cur_group_id()) %>% group_by(DOI) %>%
  mutate(sub_id = ifelse(has_sub, match(sub_key, unique(sub_key[has_sub])) + 1L, NA_integer_)) %>%
  ungroup()

exploded <- rows %>% transmute(
  path = str_c("aligns/", accession, "_", slug, ".url"),
  has_errors = FALSE,
  meta_language = "en", meta_title_raw = Name,
  meta_instrument_type_raw = InstrumentType,
  meta_publication_year_raw = as.numeric(TestYear),
  meta_authors_raw = NA_character_,
  meta_test_format_raw = NA_character_,
  meta_source_raw = url, meta_permissions_raw = Permissions, meta_language_raw = "en",
  bucket = "scaled",
  scale_id = as.numeric(coalesce(sub_id, 1L)),
  scale_name = ifelse(has_sub, scale_name, Name),
  scale_construct_name = ifelse(has_sub, scale_name, first_construct),
  scale_depth = ifelse(has_sub, 2, 1),
  item_item_id = as.numeric(item_n),
  item_item_text = item_text,
  item_has_image = FALSE, item_item_type = "rating_scale",
  item_admin_note = NA_character_, item_language = "en", item_reverse_coded = NA,
  source_url = url,
  version = str_c(meta_instrument_name, " [", database, "]"),
  saved_files = "data/aligns/larsen_instruments.parquet",
  source_grade = "public_repository", extraction_confidence = "medium",
  verbatim_claimed = TRUE, verify_verdict = NA_character_,
  retrieval_notes = str_c("Larsen/aligns corpus via ", database,
                          "; matched by name (", mtype, "); reverse coding and response options not recorded by source.")
)
exploded$scale_id_path <- map2(exploded$scale_depth, exploded$scale_id,
                               ~ if (.x == 1) 1L else c(1L, as.integer(.y)))
exploded$scale_name_path <- pmap(list(exploded$scale_depth, exploded$meta_title_raw, exploded$scale_name),
                                 function(dep, root, sub) if (dep == 1) root else c(root, sub))
exploded$item_options <- map(seq_len(nrow(exploded)), ~ character(0))

template <- arrow::read_parquet("data/restricted/scale-hunt-extractions-exploded.parquet")
for (col in setdiff(names(template), names(exploded))) exploded[[col]] <- template[[col]][NA_integer_]
exploded <- exploded %>% select(all_of(names(template)))
arrow::write_parquet(exploded, "data/aligns/aligns-extractions-exploded.parquet")
cat("aligns ingest:", nrow(exploded), "items,", n_distinct(exploded$path), "instruments\n")
write.csv(fills %>% arrange(desc(usage_count)) %>%
            select(DOI, Name, meta_instrument_name, n_items, usage_count, mtype, in_targets156),
          "data/processed/aligns_ingested.csv", row.names = FALSE)
