# Shared by semanticnet_ingest.R / aligns_ingest.R (and analysis scripts).
#
# PsycTests issues separate records for translations ("perceived stress
# scale-chinese version"). Our fill sources hold ENGLISH item text, and PsycTests
# name matching strips the suffix, so without a guard the ingests map English
# items onto translation records whenever the English original is already
# covered (the uncovered-first gate makes the translation the only candidate
# left). That misattributes the items and double-counts the instrument.
#
# A name is excluded only when a language/nationality word is followed by a
# version-type word ("chinese version", "brazilian portuguese adaptation",
# "persian short form"). A bare language word is NOT excluded: constructs like
# "african american acculturation scale" or the EORTC instruments are
# legitimate originals.
LANG_WORDS <- c(
  "chinese", "mandarin", "cantonese", "taiwanese",
  "german", "austrian", "french", "italian", "spanish", "castilian", "catalan",
  "basque", "galician", "portuguese", "brazilian", "mexican", "argentine",
  "argentinian", "chilean", "colombian", "peruvian",
  "dutch", "flemish", "danish", "norwegian", "swedish", "finnish", "icelandic",
  "japanese", "korean", "vietnamese", "thai", "khmer", "burmese", "myanmar",
  "lao", "filipino", "tagalog", "indonesian", "malay", "malaysian",
  "turkish", "persian", "farsi", "iranian", "arabic", "arab", "hebrew",
  "israeli", "egyptian", "moroccan", "tunisian", "lebanese", "jordanian",
  "saudi", "pakistani", "bangladeshi", "sri lankan",
  "urdu", "hindi", "punjabi", "tamil", "telugu", "marathi", "gujarati",
  "kannada", "bengali", "sinhala", "nepali",
  "polish", "czech", "slovak", "slovenian", "slovene", "hungarian", "romanian",
  "bulgarian", "greek", "albanian", "macedonian", "serbian", "croatian",
  "bosnian", "russian", "ukrainian", "estonian", "latvian", "lithuanian",
  "georgian", "armenian", "azerbaijani", "kazakh", "uzbek", "mongolian",
  "amharic", "ethiopian", "swahili", "yoruba", "igbo", "hausa", "zulu",
  "xhosa", "afrikaans", "maltese", "welsh", "irish"
)
LANG_VERSION_RE <- stringr::regex(stringr::str_c(
  "\\b(", paste(LANG_WORDS, collapse = "|"), ")",
  "([ -](language|speaking|canadian|portuguese|short|long|brief|revised|",
  "modified|abridged|adapted|validated|paper|electronic|[0-9]+([ -]item)?))*",
  "[ -](version|translation|translated|adaptation|adaption|edition|revision)s?\\b"
), ignore_case = TRUE)
# trailing parenthetical language: "obsessive beliefs questionnaire--child version (dutch)"
LANG_PAREN_RE <- stringr::regex(stringr::str_c(
  "\\((", paste(LANG_WORDS, collapse = "|"), ")\\)\\s*$"
), ignore_case = TRUE)
# leading language word ("chinese self-control scale"), a language word anywhere
# in a "--" variant suffix ("brief cope--swahili and kikuyu versions",
# "experiences in close relationships revised--korean"), or a language-specific
# population ("emotion regulation questionnaire for spanish adolescents"):
# in the fill sources these only ever matched via the English original's name,
# so they are translation/adaptation records, not true matches
LANG_EXTRA_RE <- stringr::regex(stringr::str_c(
  "^(", paste(LANG_WORDS, collapse = "|"), ")\\b",
  "|--[^-]*\\b(", paste(LANG_WORDS, collapse = "|"), ")\\b",
  "|\\bfor (", paste(LANG_WORDS, collapse = "|"), ")\\b"
), ignore_case = TRUE)

is_language_variant <- function(name) {
  !is.na(name) & (stringr::str_detect(name, LANG_VERSION_RE) |
                    stringr::str_detect(name, LANG_PAREN_RE) |
                    stringr::str_detect(name, LANG_EXTRA_RE))
}
