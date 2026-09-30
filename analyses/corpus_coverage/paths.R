# Local input locations for the corpus-coverage pipeline.
# Run every script from this directory (analyses/corpus_coverage/).
#
# Two inputs are NOT in the repository because they derive from the licensed
# APA PsycTests database. Put them in data/psyctests/ or point the environment
# variables at them:
#   PSYC_RECORDS  one row per PsycTests record (DOI, Name, first_acronym,
#                 instrument_type_broad, InstrumentType, TestYear, Permissions, ...)
#   PSYC_INFO     usage table (DOI, Name, Year, usage_count) derived from
#                 PsycInfo citation counts
# RETICULATE_ENV names the conda/miniconda environment that provides kaleido
# for plotly::save_image (only needed to render the treemap PNGs).

PSYC_RECORDS <- Sys.getenv("PSYC_RECORDS", "data/psyctests/preprocessed_records.rds")
PSYC_INFO <- Sys.getenv("PSYC_INFO", "data/psyctests/psyctests_info.rds")
RETICULATE_ENV <- Sys.getenv("RETICULATE_ENV", "r-reticulate-test")

for (.f in c(PSYC_RECORDS, PSYC_INFO)) {
  if (!file.exists(.f)) {
    warning("PsycTests input not found: ", .f, " (see paths.R / README.md)", call. = FALSE)
  }
}

# The Larsen/aligns instrument table is shipped as CSV; the parquet written by
# larsen/larsen_data_cleaning.Rmd is used when present (parquet files are
# git-ignored in this repository).
read_larsen_instruments <- function() {
  p <- "data/aligns/larsen_instruments.parquet"
  if (file.exists(p)) {
    arrow::read_parquet(p)
  } else {
    readr::read_csv("data/aligns/larsen_instruments.csv", show_col_types = FALSE, progress = FALSE)
  }
}
