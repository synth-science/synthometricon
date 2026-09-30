source("paths.R")
# Assemble scale-hunt extraction JSON (workflow output) into a parquet that
# reproduces the structure of data/raw-extractions-exploded.parquet so it can
# be bound to Björn's extractions. Item text is copyrighted for some scales:
# output lives in data/restricted/ (gitignored), never committed or shared.
suppressMessages({
  library(dplyr); library(tidyr); library(stringr); library(purrr)
  library(jsonlite); library(arrow)
})

assemble_scale_hunt <- function(workflow_json, out_parquet,
                                template_parquet = "data/raw-extractions-exploded.parquet") {
  template_schema <- arrow::open_dataset(template_parquet)$schema

  out <- jsonlite::fromJSON(workflow_json, simplifyVector = FALSE)
  res <- Filter(function(r) !is.null(r$extraction) && isTRUE(r$extraction$found) &&
                  length(r$extraction$items) > 0, out$result)

  psyc <- readRDS(PSYC_RECORDS) %>%
    transmute(DOI, psyc_name = Name, first_construct, psyc_instrument_type = InstrumentType,
              psyc_year = TestYear, Permissions,
              psyc_authors = sapply(AuthorList, function(a) paste(unlist(a), collapse = " | ")))

  rows <- purrr::map_dfr(res, function(r) {
    e <- r$extraction
    doi <- r$target
    meta <- psyc %>% filter(DOI == doi) %>% slice(1)
    accession <- str_c("9999", str_match(doi, "10\\.1037/t(\\d{5})-000")[, 2])
    slug <- str_replace_all(tolower(coalesce(meta$psyc_name[1], e$name, "scale")), "[^a-z0-9]+", "-")
    root_name <- coalesce(meta$psyc_name[1], e$name)

    items <- purrr::map_dfr(e$items, function(it) tibble(
      n = it$n, text = it$text %||% NA_character_,
      subscale = it$subscale %||% NA_character_,
      options_raw = it$options %||% NA_character_,
      reverse = it$reverse %||% NA,
      item_type = it$item_type %||% NA_character_,
      admin_note = it$admin_note %||% NA_character_,
      has_image = it$has_image %||% FALSE
    ))

    # batch schema: subscale construct lookup from the extraction-level subscale list
    sub_constructs <- character(0)
    for (s in e$subscales %||% list()) {
      if (!is.null(s$name)) sub_constructs[s$name] <- s$construct %||% s$name
    }

    # scale hierarchy: root scale (the instrument, depth 1) = scale_id 1;
    # named subscales become depth-2 children in order of first appearance
    sub_ids <- items %>% filter(!is.na(subscale)) %>% distinct(subscale) %>%
      mutate(sub_id = row_number() + 1L)
    items <- items %>% left_join(sub_ids, by = "subscale")

    tibble(
      path = str_c("web-retrieved/", accession, "_", slug, ".url"),
      has_errors = FALSE,
      meta_language = e$language %||% NA_character_,
      meta_title_raw = root_name,
      meta_instrument_type_raw = meta$psyc_instrument_type[1] %||% NA_character_,
      meta_publication_year_raw = as.numeric(meta$psyc_year[1] %||% NA),
      meta_authors_raw = meta$psyc_authors[1] %||% NA_character_,
      meta_test_format_raw = e$response_scale %||% NA_character_,
      meta_source_raw = e$best_source_url %||% NA_character_,
      meta_permissions_raw = meta$Permissions[1] %||% NA_character_,
      meta_language_raw = e$language %||% NA_character_,
      bucket = "scaled",
      scale_id = as.numeric(coalesce(items$sub_id, 1L)),
      scale_name = coalesce(items$subscale, root_name),
      # construct from the agent-reported subscale list when available,
      # else the subscale label itself; root items use the PsycTests construct
      scale_construct_name = coalesce(
        unname(sub_constructs[items$subscale]),
        items$subscale,
        meta$first_construct[1] %||% NA_character_
      ),
      scale_id_path = purrr::map(items$sub_id, ~ if (is.na(.x)) 1L else c(1L, .x)),
      scale_name_path = purrr::map(items$subscale, ~ if (is.na(.x)) root_name else c(root_name, .x)),
      scale_depth = ifelse(is.na(items$sub_id), 1, 2),
      item_item_id = as.numeric(items$n),
      item_item_text = items$text,
      item_has_image = items$has_image,
      # agent-reported type when present (batch schema), else derived:
      # item-specific graded options => choice, shared scale => rating_scale
      item_item_type = coalesce(items$item_type,
                                ifelse(!is.na(items$options_raw), "choice", "rating_scale")),
      item_options = purrr::map(items$options_raw,
                                ~ if (is.na(.x)) character(0) else str_split_1(.x, fixed(" || "))),
      item_admin_note = items$admin_note,
      item_language = e$language %||% NA_character_,
      item_reverse_coded = items$reverse,
      # provenance beyond Björn's schema
      source_url = e$best_source_url %||% NA_character_,
      version = e$version %||% NA_character_,
      saved_files = paste(unlist(e$saved_files %||% list()), collapse = " | "),
      source_grade = e$source_grade %||% NA_character_,
      extraction_confidence = e$confidence %||% NA_character_,
      verbatim_claimed = isTRUE(e$verbatim),
      verify_verdict = if (!is.null(r$verification)) r$verification$verdict else NA_character_,
      retrieval_notes = e$notes %||% NA_character_
    )
  })

  # add every template column we could not fill, as NA of the template's type,
  # and order columns: template order first, provenance extras last
  extras <- setdiff(names(rows), names(template_schema))
  for (f in template_schema$fields) {
    if (!f$name %in% names(rows)) {
      rows[[f$name]] <- switch(
        class(f$type)[1],
        "Int64" = NA_integer_, "Boolean" = NA,
        "LargeUtf8" = NA_character_, "Utf8" = NA_character_,
        "Null" = NA_character_,
        NA_real_
      )
    }
  }
  rows <- rows %>% select(all_of(names(template_schema)), all_of(extras))
  arrow::write_parquet(rows, out_parquet)
  rows
}

`%||%` <- function(a, b) if (is.null(a)) b else a

if (sys.nframe() == 0) {
  # assemble every result file, then bind into one combined restricted parquet.
  # NB: pilot-extractions-exploded.parquet is not regenerated here — it contains
  # the hand-rescued PANSS rows appended outside this converter.
  sources <- Sys.glob("data/restricted/batch*_salvaged.json")
  sources <- c(sources, Sys.glob("data/restricted/batch*_results.json"))
  parts <- list(arrow::read_parquet("data/restricted/pilot-extractions-exploded.parquet"))
  for (s in sources) {
    out <- sub("[.]json$", ".parquet", s)
    rows <- assemble_scale_hunt(s, out)
    cat(basename(s), ":", nrow(rows), "item rows,", n_distinct(rows$path), "instruments\n")
    parts[[length(parts) + 1]] <- rows
  }
  listcols <- c("scale_id_path", "scale_name_path", "item_options")
  parts <- lapply(parts, function(p) {
    for (lc in listcols) p[[lc]] <- lapply(p[[lc]], identity)
    p
  })
  combined <- bind_rows(parts)
  arrow::write_parquet(combined, "data/restricted/scale-hunt-extractions-exploded.parquet")
  cat("combined:", nrow(combined), "item rows,", n_distinct(combined$path), "instruments\n")
}
