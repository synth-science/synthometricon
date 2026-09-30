export const meta = {
  name: 'scale-hunt-batch',
  description: 'Batch: find verbatim item text online for missing SynthNet scales (haiku-first, sonnet rescue)',
  phases: [
    { title: 'Extract', detail: 'haiku web-hunt per scale', model: 'haiku' },
    { title: 'Rescue', detail: 'sonnet retry where haiku fell short', model: 'sonnet' },
  ],
}

const PDF_DIR = '/Users/rubenarslan/research/items_llm/data/restricted/pdfs'
const PDFINSPECT = '/Users/rubenarslan/.venvs/pdf-inspector/bin/python'

const ITEMS_SCHEMA = {
  type: 'object',
  required: ['doi', 'found', 'confidence', 'notes', 'saved_files'],
  properties: {
    doi: { type: 'string' },
    name: { type: 'string' },
    version: { type: ['string', 'null'], description: 'which version/form was extracted, e.g. "PSS-14 original 1983", "parent form"' },
    found: { type: 'boolean' },
    best_source_url: { type: ['string', 'null'] },
    other_source_urls: { type: 'array', items: { type: 'string' } },
    saved_files: { type: 'array', items: { type: 'string' }, description: 'absolute paths of source files you archived under the pdfs dir' },
    source_grade: { type: ['string', 'null'], enum: ['official', 'journal_appendix', 'university_pdf', 'clinical_org', 'other_web', null] },
    language: { type: ['string', 'null'] },
    response_scale: { type: ['string', 'null'] },
    subscales: { type: 'array', items: { type: 'object', required: ['name'], properties: { name: { type: 'string' }, construct: { type: ['string', 'null'], description: 'construct measured, if stated or obvious (e.g. subscale "PT" -> "Perspective taking")' } } } },
    items: {
      type: 'array',
      items: {
        type: 'object',
        required: ['n', 'text'],
        properties: {
          n: { type: 'integer' },
          text: { type: 'string', description: 'verbatim item text exactly as in the archived source' },
          subscale: { type: ['string', 'null'] },
          options: { type: ['string', 'null'], description: 'item-specific response options verbatim, joined with " || "; null if shared scale' },
          reverse: { type: ['boolean', 'null'] },
          item_type: { type: ['string', 'null'], enum: ['rating_scale', 'choice', 'open', 'other', null] },
          admin_note: { type: ['string', 'null'], description: 'administration/rating note or brief item definition (e.g. interviewer rating basis)' },
          has_image: { type: ['boolean', 'null'] },
        },
      },
    },
    extracted_item_count: { type: 'integer' },
    verbatim: { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    notes: { type: 'string' },
  },
}

function extractPrompt(t) {
  return `You are hunting for the VERBATIM item text of a psychological instrument for a legitimate psychometrics research project (computing averaged vector embeddings for a library search; items stored locally only, never redistributed — fair use). Any good public web source is acceptable: official instrument sites, journal appendices, author-archived PDFs, university course materials, clinical org sites, measure repositories. Do NOT bypass logins or solve CAPTCHAs.

TARGET INSTRUMENT
- Name: ${t.name}${t.acronym ? ' (' + t.acronym + ')' : ''}
- PsycTests DOI: ${t.doi} | accession: ${t.accession}
- Alternate names: ${t.alt_names || 'none'}
- Authors: ${t.authors || 'unknown'}
- Original publication: ${t.source_citation || 'unknown'}
- Official website(s): ${t.websites || 'none listed'}
- Instrument type: ${t.instrument_type || 'unknown'}
- Expected number of items (metadata, may be wrong): ${t.expected_items ?? 'unknown'}
- Expected subscales: ${t.subscale_info || t.expected_subscales || 'unknown'}

TOOLS: ToolSearch "select:WebFetch" first. NOTE: the built-in WebSearch quota is exhausted this session — do NOT call WebSearch. Discovery instead, in this order:
1. DuckDuckGo via curl: curl -sL "https://html.duckduckgo.com/html/?q=<url-encoded query>" | grep -o "href=\\"[^\\"]*\\"" — parse result URLs, then WebFetch/curl the promising ones. Queries: name/acronym + "items", "appendix", "questionnaire pdf".
2. VPN-Paper-Search public-index searches (plain public APIs, NOT VPN traffic): search_openalex, search_crossref, search_semantic — up to 4 calls per scale; also get_paper WITHOUT allow_institutional (open-access tiers only) to pull an OA paper PDF by DOI, once.
3. Direct WebFetch/curl of likely hosts (official site from metadata, clinical/university repositories guessed from the citation).
FORBIDDEN always: fetch_paywalled, fetch_paywalled_batch, login_to, vpn_switch, download_witten_* — these switch the machine VPN or open interactive browser windows and must never run inside batch agents. Zotero tools may also hold the original paper.

EFFORT BUDGET (hard rule): at most ~12 tool calls and ~4 source fetches in total. If two sources in a row fail to yield the items, STOP hunting — return your best partial with honest notes (a later pass handles leftovers). Do not cross-check extra sources once one good source is archived.

MANDATORY SOURCE ARCHIVING — a deterministic script will later verify every extracted item appears verbatim in your archived files, so archive EXACTLY what you extract from:
- PDFs: curl -sL -o "${PDF_DIR}/${t.accession}_${t.slug}_<shortsource>.pdf" "<url>"
- HTML pages used as an item source: curl -sL -o "${PDF_DIR}/${t.accession}_${t.slug}_<shortsource>.html" "<url>"
List every archived file in saved_files (absolute paths).

READING PDFs (cheap route first): after downloading, run
  ${PDFINSPECT} -c "import pdf_inspector,sys; open(sys.argv[2],'w',encoding='utf-8').write(pdf_inspector.process_pdf(sys.argv[1]).markdown)" "<pdf>" "<pdf>.md"
then Read the .md file. If that errors or yields empty/garbled text (scanned PDF), Read the PDF directly with the pages parameter.

TASK
1. Find the best source with the FULL item list (official > journal appendix > university PDF > clinical org > other). Up to ~6 fetches; return best partial with honest notes if the full list is unfindable.
2. Extract EVERY item verbatim with original numbering. NEVER reconstruct from memory or paraphrase — only text present in an archived source from this session. If you cannot access any source, found=false.
3. Record version explicitly (short forms, revisions, informant forms differ).
4. Subscale assignment where source or metadata makes it unambiguous; list subscales with the construct each measures when stated.
5. Per item: item_type (rating_scale/choice/open/other), reverse coding if stated, item-specific options verbatim in options, rater instructions or brief definitions in admin_note.
6. LICENSED CLINICAL INSTRUMENTS with long per-level anchor definitions (e.g. 7-point anchored rating manuals) or very large copyrighted inventories: put item stems/names + a one-sentence definition in your output, NOT the full multi-level anchor text; the full text stays in your archived files. Never reproduce more than ~2 pages of licensed manual text in your response — oversized verbatim output gets blocked by a content filter and your whole run is lost.
7. For interview schedules without discrete item text, extract the itemized probe/symptom list if one exists; otherwise found=false with notes.

Your final answer is consumed by a script: return ONLY the structured output.`
}

const targets = typeof args === 'string' ? JSON.parse(args) : args
log(`Batch of ${targets.length} scales (haiku first, sonnet rescue)`)

function needsRescue(e, t) {
  // cost-limited: rescue only when the haiku pass produced nothing usable
  if (!e) return true
  if (!e.found || !e.items || e.items.length === 0) return true
  return false
}

const results = await pipeline(
  targets,
  t => agent(extractPrompt(t), { label: `ex:${t.acronym || t.name.slice(0, 20)}`, phase: 'Extract', model: 'haiku', effort: 'low', schema: ITEMS_SCHEMA }),
  (e, t) => {
    if (!needsRescue(e, t)) return { target: t.doi, tier: 'haiku', extraction: e }
    const prior = e
      ? `NOTE: a first attempt by a smaller model returned found=${e.found}, ${e.items ? e.items.length : 0} items, confidence=${e.confidence}. Its notes: ${(e.notes || '').slice(0, 300)}. Do better — find and archive a solid source.`
      : 'NOTE: a first attempt died with an API error, possibly the content filter on bulk licensed text. Keep item output compact (stems + one-sentence definitions only, full text stays in archived files).'
    return agent(prior + '\n\n' + extractPrompt(t), { label: `rescue:${t.acronym || t.name.slice(0, 20)}`, phase: 'Rescue', model: 'sonnet', effort: 'low', schema: ITEMS_SCHEMA })
      .then(r => ({ target: t.doi, tier: 'sonnet-rescue', extraction: r, haiku_found: e ? !!e.found : null }))
  }
)
const done = results.filter(Boolean)
const found = done.filter(r => r.extraction && r.extraction.found)
const rescued = done.filter(r => r.tier === 'sonnet-rescue')
log(`${found.length}/${targets.length} found (${rescued.length} needed sonnet rescue)`)
return results