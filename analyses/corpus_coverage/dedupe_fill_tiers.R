source("paths.R")
# Content-level deduplication of the fill tiers, for the SynthNet search engine.
# Run AFTER scale_hunt_assemble.R, semanticnet_ingest.R and aligns_ingest.R:
# it rewrites the three fill parquets in place.
#
# DOI-level disjointness across tiers does not prevent the SAME instrument
# appearing twice under sibling PsycTests records (e.g. the hunt's SCL-90-R
# next to SynthNet's "self-report symptom inventory") — near-identical item
# sets that would surface as duplicate results in the search engine.
#
# Rule: two records are duplicates when the Jaccard similarity of their
# normalised item texts is >= 0.8. Containment alone (short form inside its
# long form, PHQ-9 inside PHQ) does NOT count: short forms are distinct
# instruments and stay. SynthNet records are always kept (Björn's corpus is not
# ours to filter). Fill records are visited by item count (ties by tier: hunt >
# semanticnet > aligns) and dropped only when they are a direct duplicate
# (J >= 0.8) of a record already kept; a record linked to the corpus only
# through another dropped record is kept. Dropped records are logged to
# data/processed/fill_tier_dedupe_aliases.csv (dropped DOI -> kept DOI), which
# scale_hunt_treemap.R reads so aliased DOIs still count as covered — their
# content is in the corpus under the kept record.
suppressMessages({
  library(dplyr); library(stringr); library(arrow); library(purrr)
})

J_THRESHOLD <- 0.8
FILL_FILES <- c(hunt = "data/restricted/scale-hunt-extractions-exploded.parquet",
                semanticnet = "data/semanticnet/semanticnet-extractions-exploded.parquet",
                aligns = "data/aligns/aligns-extractions-exploded.parquet")
TIER_RANK <- c(synthnet = 0, hunt = 1, semanticnet = 2, aligns = 3)

doi_from_path <- function(p) str_c("10.1037/t", str_sub(str_extract(basename(p), "[0-9]{9}"), 5, 9), "-000")
norm_item <- function(x) tolower(str_squish(str_replace_all(x, "[^A-Za-z0-9 ]", " ")))

items_of <- function(df, tier) df %>%
  transmute(tier = tier, DOI = doi_from_path(path), item = norm_item(item_item_text)) %>%
  filter(nchar(item) >= 12) %>% distinct(tier, DOI, item)

fill <- imap(FILL_FILES, ~ read_parquet(.x))
all_items <- bind_rows(
  bind_rows(imap(fill, ~ items_of(.x, .y))),
  open_dataset("data/raw-extractions-exploded.parquet") %>% filter(bucket == "scaled") %>%
    select(path, item_item_text) %>% collect() %>% items_of("synthnet"))
sizes <- all_items %>% count(tier, DOI, name = "n")

pairs <- all_items %>%
  group_by(item) %>% filter(n_distinct(str_c(tier, DOI)) > 1) %>% ungroup() %>%
  inner_join(., ., by = "item", relationship = "many-to-many") %>%
  filter(str_c(tier.x, DOI.x) < str_c(tier.y, DOI.y),
         tier.x != "synthnet" | tier.y != "synthnet") %>%
  count(tier.x, DOI.x, tier.y, DOI.y, name = "shared") %>%
  left_join(sizes, by = c("tier.x" = "tier", "DOI.x" = "DOI")) %>% rename(nx = n) %>%
  left_join(sizes, by = c("tier.y" = "tier", "DOI.y" = "DOI")) %>% rename(ny = n) %>%
  mutate(J = shared / (nx + ny - shared)) %>%
  filter(J >= J_THRESHOLD)

# Greedy over every record that has a duplicate edge: SynthNet records first (always kept),
# then fill records by item count and tier. A fill record is dropped only if it has a DIRECT
# duplicate edge (J >= .8) to a record already kept, and is aliased to the most similar such
# record. Records linked only through a chain of dropped records (A ~ B ~ SynthNet with
# J(A, SynthNet) < .8) are therefore kept, and no two kept fill records are duplicates.
key <- function(t, d) str_c(t, "|", d)
edges <- bind_rows(
  pairs %>% transmute(a = key(tier.x, DOI.x), b = key(tier.y, DOI.y), J),
  pairs %>% transmute(a = key(tier.y, DOI.y), b = key(tier.x, DOI.x), J))
groups <- tibble(node = unique(edges$a)) %>%
  tidyr::separate(node, c("tier", "DOI"), sep = "\\|", remove = FALSE) %>%
  left_join(sizes, by = c("tier", "DOI")) %>%
  arrange(TIER_RANK[tier] != 0, desc(n), TIER_RANK[tier])

kept_keys <- character(0)
alias <- list()
for (i in seq_len(nrow(groups))) {
  g <- groups[i, ]
  if (g$tier == "synthnet") { kept_keys <- c(kept_keys, g$node); next }
  dup <- edges %>% filter(a == g$node, b %in% kept_keys) %>% arrange(desc(J)) %>% slice_head(n = 1)
  if (nrow(dup)) {
    alias[[length(alias) + 1]] <- tibble(tier = g$tier, DOI = g$DOI, n = g$n, best = dup$b, J = dup$J)
  } else {
    kept_keys <- c(kept_keys, g$node)
  }
}
alias <- bind_rows(alias, tibble(tier = character(), DOI = character(), n = integer(),
                                 best = character(), J = double())) %>%
  tidyr::separate(best, c("kept_tier", "kept_DOI"), sep = "\\|")

psyc <- readRDS(PSYC_RECORDS) %>% select(DOI, Name)
alias_out <- alias %>%
  left_join(psyc, by = "DOI") %>% rename(dropped_name = Name) %>%
  left_join(psyc, by = c("kept_DOI" = "DOI")) %>% rename(kept_name = Name) %>%
  select(dropped_tier = tier, dropped_DOI = DOI, dropped_name, n_items = n,
         kept_tier, kept_DOI, kept_name, J)
write.csv(alias_out, "data/processed/fill_tier_dedupe_aliases.csv", row.names = FALSE)

for (t in names(FILL_FILES)) {
  drop_dois <- alias_out %>% filter(dropped_tier == t) %>% pull(dropped_DOI)
  if (!length(drop_dois)) { cat(t, ": nothing to drop\n"); next }
  df <- fill[[t]]
  before <- c(nrow(df), n_distinct(df$path))
  df <- df %>% filter(!doi_from_path(path) %in% drop_dois)
  write_parquet(df, FILL_FILES[[t]])
  cat(sprintf("%s: dropped %d records (%d -> %d items, %d -> %d instruments)\n",
              t, length(drop_dois), before[1], nrow(df), before[2], n_distinct(df$path)))
}
cat("\naliases written to data/processed/fill_tier_dedupe_aliases.csv:\n")
print(alias_out %>% transmute(dropped_tier, dropped = str_trunc(coalesce(dropped_name, dropped_DOI), 42),
                              kept_tier, kept = str_trunc(coalesce(kept_name, kept_DOI), 42)) %>%
        as.data.frame(), right = FALSE)
