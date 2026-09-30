export const meta = {
  name: 'scale-hunt-batch09',
  description: 'Batch09: last viable leftovers — sonnet-first hunt with WebSearch restored and per-scale hints',
  phases: [
    { title: 'Extract', detail: 'sonnet web-hunt per scale with hints', model: 'sonnet' },
    { title: 'Rescue', detail: 'high-effort sonnet retry where the first pass fell short', model: 'sonnet' },
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
    version: { type: ['string', 'null'], description: 'which version/form was extracted, e.g. "CTS Form N 1979", "parent form"' },
    found: { type: 'boolean' },
    best_source_url: { type: ['string', 'null'] },
    other_source_urls: { type: 'array', items: { type: 'string' } },
    saved_files: { type: 'array', items: { type: 'string' }, description: 'absolute paths of source files you archived under the pdfs dir' },
    source_grade: { type: ['string', 'null'], enum: ['official', 'journal_appendix', 'university_pdf', 'clinical_org', 'other_web', null] },
    language: { type: ['string', 'null'] },
    response_scale: { type: ['string', 'null'] },
    subscales: { type: 'array', items: { type: 'object', required: ['name'], properties: { name: { type: 'string' }, construct: { type: ['string', 'null'], description: 'construct measured, if stated or obvious' } } } },
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
          admin_note: { type: ['string', 'null'], description: 'administration/rating note or brief item definition' },
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

const HINTS = {
  '10.1037/t02125-000': 'PRIOR ATTEMPT: died to a transient API error, NOT a retrieval failure — the items are findable. The original CTS item list (Form N, ~19 conflict-tactic items in escalating aggression order) is in the appendix of Straus 1979, Journal of Marriage and the Family 41(1):75-88 (doi 10.2307/351733); PDFs are hosted on university sites (e.g. UNH Family Research Lab pubpages) and the items are reproduced in many public theses. Keep per-item output compact.',
  '10.1037/t07870-000': 'Interview schedule (Endicott & Spitzer 1978, Archives of General Psychiatry 35(7):837-844). The full SADS interview booklet has circulated as PDF from university/clinical sites (Columbia/NYSPI heritage). Extract the itemized symptom rating list: item stem + one-sentence rating basis, NOT full multi-level anchors (content filter risk). A substantial partial (e.g. the mood/psychosis rating items) is valuable.',
  '10.1037/t73087-000': 'PAR-commercial (86 items, parent/teacher forms, scales: Inhibit, Shift, Emotional Control, Initiate, Working Memory, Plan/Organize, Organization of Materials, Monitor). Full item text rarely public — hunt open-access dissertations/theses with item-level tables (factor loadings with item text), or translated validation articles listing all items in English. Record which form. Partial item sets are worth returning.',
  '10.1037/t02278-000': 'Target the ORIGINAL IIP-127 (Horowitz et al. 1988, JCCP 56(6):885-892). Mind Garden sells IIP-64/IIP-32 — skip official sources. Public dissertations and older validation articles often reproduce the full 127-item list or the IIP-64 subset; either is valuable (record version precisely). Items have two stem formats: "It is hard for me to..." and "...too much" statements.',
  '10.1037/t55271-000': 'Thai instrument: Sakunpong, Choochom & Taephant 2016, "Development of a resilience scale for Thai substance-dependent women", Asian Journal of Psychiatry 22:177-181, doi 10.1016/j.ajp.2015.10.011. Elsevier paywalled — try OA mirrors, Thai university repositories (Srinakharinwirot BSRI, Chulalongkorn), ThaiJO journals, or the authors\' related theses. Items likely in Thai (language=th, fine) — the article may list example items only; a related Thai thesis may hold all 71.',
  '10.1037/t41553-000': 'Hamilton Depression Inventory (Reynolds & Kobak 1995, Psychological Assessment 7(4):472-483) — PAR-commercial 23-item self-report adaptation of the HAM-D. Full text rarely public; hunt OA dissertations (ProQuest open copies, university repositories) that reproduce items or item stems. Do NOT confuse with the clinician HAM-D (already covered). Item stems + brief definitions suffice.',
  '10.1037/t01309-000': 'Arnold, O\'Leary, Wolff & Acker 1993, Psychological Assessment 5(2):137-144. 30 items; each item is a discipline situation stem with TWO opposing 7-point anchor statements (effective vs ineffective). The scale PDF is freely hosted by child-clinic and university sites (search "Parenting Scale" Arnold O\'Leary pdf). Put the situation stem in text and both anchors in options.',
  '10.1037/t06497-000': 'Spielberger STAIC ("How I Feel Questionnaire", 1973, Mind Garden commercial): 20 state items ("I feel... calm/upset/...") + 20 trait items. Public dissertations and older validation/translation articles often reproduce one or both forms. Record which form(s); one full form is a win.',
  '10.1037/t05183-000': 'Single-item 0-100 self-rating (Wolpe, Subjective Units of Disturbance/Distress). Two candidate sources are already in metadata websites: an Oxford Clinical Psychology appendix and an inneractions.com.au PDF with anchor descriptions. Return 1 item: the rating question/instruction as text, anchor levels verbatim in options.',
  '10.1037/t09699-000': 'Kohn & Rosman 1972, Developmental Psychology 6(3):430-444 (doi 10.1037/h0032583): 58-item preschool Symptom Checklist (factors: Apathy-Withdrawal, Anger-Defiance). Item text may be in the article tables/appendix — try OA copies, ERIC microfiche-era documents (Kohn Problem Checklist), and later validation papers. The related "Kohn Problem Checklist" listing is an acceptable source if it is the same instrument (note it).',
}

function extractPrompt(t) {
  return `You are hunting for the VERBATIM item text of a psychological instrument for a legitimate psychometrics research project (computing averaged vector embeddings for a library search; items stored locally only, never redistributed — fair use). Any good public web source is acceptable: official instrument sites, journal appendices, author-archived PDFs, university course materials, clinical org sites, measure repositories. Do NOT bypass logins or solve CAPTCHAs.

THIS IS A FINAL-PASS TARGET: earlier hunts missed it. Scale-specific intel:
${HINTS[t.doi] || 'none'}

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

TOOLS: ToolSearch "select:WebFetch,WebSearch" first — BOTH work this session. Discovery, in this order:
1. WebSearch (quota is back): up to 5 searches per scale. Queries: name/acronym + "items", "appendix", "questionnaire pdf", author + year, site:*.edu / repository terms.
2. DuckDuckGo via curl as fallback: curl -sL "https://html.duckduckgo.com/html/?q=<url-encoded query>" | grep -o "href=\\"[^\\"]*\\"" — parse result URLs, then WebFetch/curl the promising ones.
3. VPN-Paper-Search public-index searches (plain public APIs, NOT VPN traffic): search_openalex, search_crossref, search_semantic — up to 4 calls per scale; also get_paper WITHOUT allow_institutional (open-access tiers only) to pull an OA paper PDF by DOI, once.
4. Direct WebFetch/curl of likely hosts (official site from metadata, clinical/university repositories guessed from the citation).
FORBIDDEN always: fetch_paywalled, fetch_paywalled_batch, login_to, vpn_switch, download_witten_* — these switch the machine VPN or open interactive browser windows and must never run inside batch agents.

EFFORT BUDGET (hard rule): at most ~16 tool calls and ~6 source fetches in total. If three sources in a row fail to yield the items, STOP hunting — return your best partial with honest notes. Do not cross-check extra sources once one good source is archived.

MANDATORY SOURCE ARCHIVING — a deterministic script will later verify every extracted item appears verbatim in your archived files, so archive EXACTLY what you extract from:
- PDFs: curl -sL -o "${PDF_DIR}/${t.accession}_${t.slug}_<shortsource>.pdf" "<url>"
- HTML pages used as an item source: curl -sL -o "${PDF_DIR}/${t.accession}_${t.slug}_<shortsource>.html" "<url>"
List every archived file in saved_files (absolute paths).

READING PDFs (cheap route first): after downloading, run
  ${PDFINSPECT} -c "import pdf_inspector,sys; open(sys.argv[2],'w',encoding='utf-8').write(pdf_inspector.process_pdf(sys.argv[1]).markdown)" "<pdf>" "<pdf>.md"
then Read the .md file. If that errors or yields empty/garbled text (scanned PDF), Read the PDF directly with the pages parameter.

TASK
1. Find the best source with the FULL item list (official > journal appendix > university PDF > clinical org > other). Return best partial with honest notes if the full list is unfindable.
2. Extract EVERY item verbatim with original numbering. NEVER reconstruct from memory or paraphrase — only text present in an archived source from this session. If you cannot access any source, found=false.
3. Record version explicitly (short forms, revisions, informant forms differ).
4. Subscale assignment where source or metadata makes it unambiguous; list subscales with the construct each measures when stated.
5. Per item: item_type (rating_scale/choice/open/other), reverse coding if stated, item-specific options verbatim in options, rater instructions or brief definitions in admin_note.
6. LICENSED CLINICAL INSTRUMENTS with long per-level anchor definitions or very large copyrighted inventories: put item stems/names + a one-sentence definition in your output, NOT the full multi-level anchor text; the full text stays in your archived files. Never reproduce more than ~2 pages of licensed manual text in your response — oversized verbatim output gets blocked by a content filter and your whole run is lost.
7. For interview schedules without discrete item text, extract the itemized probe/symptom list if one exists; otherwise found=false with notes.

Your final answer is consumed by a script: return ONLY the structured output.`
}

const targets = typeof args === 'string' ? JSON.parse(args) : args
log(`Batch09: ${targets.length} final-pass scales (sonnet first, high-effort rescue)`)

function needsRescue(e) {
  if (!e) return true
  if (!e.found || !e.items || e.items.length === 0) return true
  return false
}

const results = await pipeline(
  targets,
  t => agent(extractPrompt(t), { label: `ex:${t.acronym || t.name.slice(0, 20)}`, phase: 'Extract', model: 'sonnet', effort: 'medium', schema: ITEMS_SCHEMA }),
  (e, t) => {
    if (!needsRescue(e)) return { target: t.doi, tier: 'sonnet', extraction: e }
    const prior = e
      ? `NOTE: a first attempt returned found=${e.found}, ${e.items ? e.items.length : 0} items, confidence=${e.confidence}. Its notes: ${(e.notes || '').slice(0, 400)}. Try DIFFERENT sources/queries than those notes describe.`
      : 'NOTE: a first attempt died with an API error, possibly the content filter on bulk licensed text. Keep item output compact (stems + one-sentence definitions only, full text stays in archived files).'
    return agent(prior + '\n\n' + extractPrompt(t), { label: `rescue:${t.acronym || t.name.slice(0, 20)}`, phase: 'Rescue', model: 'sonnet', effort: 'high', schema: ITEMS_SCHEMA })
      .then(r => ({ target: t.doi, tier: 'sonnet-rescue', extraction: r, first_found: e ? !!e.found : null }))
  }
)
const done = results.filter(Boolean)
const found = done.filter(r => r.extraction && r.extraction.found)
const rescued = done.filter(r => r.tier === 'sonnet-rescue')
log(`${found.length}/${targets.length} found (${rescued.length} needed rescue)`)
return results