source("paths.R")
# Ingest SemanticNet items (Rosenbusch et al., non-proprietary) for PsycTests
# records not yet covered by SynthNet (bucket "scaled") or the scale hunt.
# Gate: name-distinctive match AND item count consistent with PsycTests metadata.
# Output mirrors Björn's exploded schema; lives in data/semanticnet/ (shareable source).
suppressMessages({
  library(dplyr); library(tidyr); library(stringr); library(purrr); library(arrow)
})

source("lang_exclude.R")

items <- read.csv("data/semanticnet/items_clean.csv") %>%
  mutate(scale = str_squish(tolower(scale)))
# Scales whose name does not identify a published instrument are not matched at all:
# - IPIP proxy keys (ipip.ori.org "new<Inventory>Key.htm", "BFASKeys.htm"): facet scales named
#   after constructs of other inventories ("insight", "self-efficacy");
# - survey archives (ARDA "thearda", the LISS panel): module labels such as "coping" or "drug use";
# - scales whose best name match turned out to be a different instrument on manual review
#   (data/processed/semanticnet_scale_exclusions.csv, with reasons).
NOT_INSTRUMENT_RE <- regex("keys?\\.htm|^thearda$|^liss$", ignore_case = TRUE)
excluded_scales <- union(
  items %>% filter(str_detect(str_trim(origin), NOT_INSTRUMENT_RE)) %>% distinct(scale) %>% pull(scale),
  read.csv("data/processed/semanticnet_scale_exclusions.csv")$semanticnet_scale)
m <- read.csv("data/semanticnet/scale_matches.csv") %>% filter(match_type != "none", dois != "")
doi_of <- function(path) str_c("10.1037/t", str_sub(str_extract(path, "[0-9]{9}"), 5, 9), "-000")
# coverage of the earlier tiers, read live: the extraction (bucket "scaled") and the hunt parquet
synthnet_dois <- arrow::open_dataset("data/raw-extractions-exploded.parquet") %>%
  filter(bucket == "scaled") %>% distinct(path) %>% collect() %>% mutate(DOI = doi_of(path)) %>%
  pull(DOI) %>% unique()
hunt_dois <- arrow::read_parquet("data/restricted/scale-hunt-extractions-exploded.parquet") %>%
  distinct(path) %>% mutate(DOI = doi_of(path)) %>% pull(DOI) %>% unique()
p <- readRDS(PSYC_RECORDS)
meta <- p %>% transmute(DOI, Name, first_construct, InstrumentType, TestYear, Permissions,
                        expected = number_of_test_items_best_guess)
pi <- readRDS(PSYC_INFO)
usage <- pi %>% group_by(DOI) %>% summarise(usage_count = sum(usage_count, na.rm = TRUE))
in_targets <- read.csv("data/semanticnet/psyc_coverage_flags.csv") %>% select(DOI, in_targets156)

# names without generic words ("scale", "inventory", ...; as in aligns_ingest.R) and without
# bare numbers ("trauma symptom checklist-40", "affectometer 2"); roman numerals are kept,
# they mark versions with different items (self-description questionnaire i vs iii)
norm <- function(x) str_squish(str_replace_all(tolower(x), "[^a-z0-9 ]", " "))
GENERIC <- c("scale","scales","questionnaire","inventory","test","index","survey","measure",
             "checklist","schedule","form","revised","short","brief","version","assessment",
             "interview","rating","the")
core <- function(x) sapply(str_split(norm(x), " "), function(t) paste(setdiff(t[!str_detect(t, "^[0-9]+$")], GENERIC), collapse = " "))

# Candidate pairs must be compatible: a short scale name (< 3 words and < 20 characters) is
# only trusted when it equals the record name apart from generic words and numbers
# ("perceived stress" = "perceived stress scale", but not "self-efficacy questionnaire for
# children"), and the item count must be within 25% of the record's documented count.
cand <- m %>% filter(!semanticnet_scale %in% excluded_scales) %>%
  separate_rows(dois, sep = ";") %>% rename(DOI = dois) %>%
  left_join(in_targets, by = "DOI") %>% left_join(meta, by = "DOI") %>%
  left_join(usage, by = "DOI") %>%
  mutate(usage_count = coalesce(usage_count, 0),
         ntok = str_count(semanticnet_scale, "[[:alnum:]]+"),
         core_eq = nchar(core(semanticnet_scale)) > 0 & core(semanticnet_scale) == core(Name),
         name_safe = ntok >= 3 | nchar(semanticnet_scale) >= 20 | core_eq,
         count_ok = is.na(expected) | (abs(n_items - expected) / pmax(expected, 1)) <= 0.25,
         covered_synthnet = DOI %in% synthnet_dois,
         uncovered = !covered_synthnet & !(DOI %in% hunt_dois)) %>%
  filter(name_safe, count_ok)

# Each SemanticNet scale goes to its best-matching record among ALL compatible records (most
# used; ties: closest item count, then DOI), one scale per record (most items). If that record is
# a translation (SemanticNet items are English) or already covered by an earlier tier, the
# scale fills nothing: it must not fall through to a lesser, uncovered match.
best <- cand %>%
  arrange(desc(usage_count), abs(n_items - expected), DOI) %>%
  group_by(semanticnet_scale) %>% slice_head(n = 1) %>% ungroup() %>%
  filter(!is_language_variant(Name)) %>%
  group_by(DOI) %>% slice_max(n_items, n = 1, with_ties = FALSE) %>% ungroup()

# full matched set, regardless of coverage by other tiers (for coverage_by_source.R, which
# reports overlapping coverage per source)
best %>%
  transmute(DOI, Name, semanticnet_scale, n_items, usage_count, match_type,
            covered_synthnet, covered_hunt = DOI %in% hunt_dois) %>%
  write.csv("data/processed/semanticnet_matches_all.csv", row.names = FALSE)

fills <- best %>% filter(uncovered)

rows <- fills %>%
  mutate(accession = str_c("9999", str_match(DOI, "10[.]1037/t([0-9]{5})-000")[, 2]),
         slug = str_sub(str_replace_all(tolower(Name), "[^a-z0-9]+", "-"), 1, 50)) %>%
  left_join(items %>% rename(semanticnet_scale = scale), by = "semanticnet_scale",
            relationship = "many-to-many") %>%
  group_by(DOI) %>% mutate(item_n = row_number()) %>% ungroup()

exploded <- rows %>% transmute(
  path = str_c("semanticnet/", accession, "_", slug, ".url"),
  has_errors = FALSE,
  meta_language = "en", meta_title_raw = Name,
  meta_instrument_type_raw = InstrumentType,
  meta_publication_year_raw = as.numeric(TestYear),
  meta_authors_raw = NA_character_,
  meta_test_format_raw = NA_character_,
  meta_source_raw = origin, meta_permissions_raw = Permissions, meta_language_raw = "en",
  bucket = "scaled",
  scale_id = 1, scale_name = Name, scale_construct_name = first_construct,
  scale_depth = 1,
  item_item_id = as.numeric(item_n),
  item_item_text = item,
  item_has_image = FALSE, item_item_type = "rating_scale",
  item_admin_note = NA_character_, item_language = "en", item_reverse_coded = NA,
  source_url = origin, version = "semanticnet (version/form not recorded)",
  saved_files = "data/semanticnet/items_clean.csv",
  source_grade = "semanticnet", extraction_confidence = "medium",
  verbatim_claimed = TRUE, verify_verdict = NA_character_,
  retrieval_notes = str_c("SemanticNet DB (Rosenbusch et al.), matched via '", semanticnet_scale,
                          "' (", match_type, "); text lowercased/punctuation-stripped by source; no subscale/options/reverse metadata.")
)
exploded$scale_id_path <- lapply(seq_len(nrow(exploded)), function(i) 1L)
exploded$scale_name_path <- lapply(exploded$scale_name, function(x) x)
exploded$item_options <- lapply(seq_len(nrow(exploded)), function(i) character(0))

template <- arrow::read_parquet("data/restricted/scale-hunt-extractions-exploded.parquet")
for (col in setdiff(names(template), names(exploded))) exploded[[col]] <- template[[col]][NA_integer_]
exploded <- exploded %>% select(all_of(names(template)))
arrow::write_parquet(exploded, "data/semanticnet/semanticnet-extractions-exploded.parquet")
cat("semanticnet ingest:", nrow(exploded), "items,", n_distinct(exploded$path), "instruments\n")
write.csv(fills %>% select(DOI, Name, semanticnet_scale, n_items, usage_count, in_targets156, match_type),
          "data/processed/semanticnet_ingested.csv", row.names = FALSE)
