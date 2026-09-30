#!/usr/bin/env python3
"""Rebuild data/semanticnet/items_clean.csv from the raw SemanticNet export.

The raw export (`items(in).csv`, from Rosenbusch et al.) is malformed CSV in
four independent ways; a naive read lumps scale names, item text, citations and
origins together. This script reconstructs the logical records deterministically:

1. Lines end in '\\r\\r\\n' and MOST (not all) carry a ';;' suffix. A record may
   span several physical lines, so lines are rejoined until the record's two
   empty trailing fields appear. Splitting on ';;\\r\\r\\n' instead would glue
   every record inside a ';;'-free block into a single row.
2. When a field contained a newline, the exporter CLOSED the quote at the line
   break and REOPENED it on the continuation line. Those seam quotes are
   artifacts and must be dropped when rejoining.
3. Many records are double-wrapped: the whole record sits inside one pair of
   outer quotes with inner quotes escaped ("scale, item,""citation"",,"). Parsed
   naively this yields a single field, which is how citations ended up glued to
   item text (e.g. the Dobrow & Tosti-Kharas "calling" scale). In records that
   also spanned lines, the citation's quoting is destroyed outright, so the
   origin field is located by what it looks like (APA year, or a URL).
4. The file is cp1252, not latin-1. Decoding it as latin-1 turns em dashes into
   U+0097 control characters ("a musician\\u0097either professionally...").

Scale names may themselves contain unquoted commas ("lesbian, gay, bisexual
identity"), so plain URL-origin records are anchored from the RIGHT: the last
four fields are always item, origin, '', ''.

Usage: semanticnet_clean.py [--check]   (--check compares against the current file)
"""
import csv, io, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data/semanticnet/items(in).csv")
OUT = os.path.join(BASE, "data/semanticnet/items_clean.csv")

# bytes undefined in cp1252 -> their windows "best fit" / obvious intent
UNDEFINED = {0x81: "", 0x8d: "", 0x8f: "", 0x90: "", 0x9d: ""}


def decode(path):
    data = open(path, "rb").read()
    out = []
    for b in data:
        if b in UNDEFINED:
            out.append(UNDEFINED[b])
        else:
            out.append(bytes([b]).decode("cp1252"))
    return "".join(out)


def is_complete(acc):
    """A logical record always ends with the two empty trailing fields."""
    a = acc.rstrip()
    return a.endswith(",,") or a.endswith(',,"')


def logical_records(text):
    """Split on the physical-line terminator, rejoin continuation lines.

    The true terminator is '\\r\\r\\n'; ';;' is an OPTIONAL suffix the exporter
    added to some lines. Splitting on ';;\\r\\r\\n' instead silently glues every
    record in a ';;'-free block into one row.
    """
    recs, acc = [], ""
    for chunk in text.split("\r\r\n"):
        chunk = chunk[:-2] if chunk.rstrip().endswith(";;") else chunk
        # a bare CR/LF can also sit inside a field (e.g. a wrapped scale name)
        line = re.sub(r"[\r\n]+", " ", chunk).strip()
        line = line[:-2].strip() if line.endswith(";;") else line
        if not acc:
            if not line:
                continue
            acc = line
        else:
            if not line:
                continue
            # drop the exporter's artificial close/reopen quotes at the seam
            acc = acc[:-1] if acc.endswith('"') else acc
            line = line[1:] if line.startswith('"') else line
            acc = acc + " " + line
        if is_complete(acc):
            recs.append(acc)
            acc = ""
    if acc.strip():
        recs.append(acc)
    return recs


CITE_SPLIT = re.compile(r',\s*""')
# start of an APA author list: ", Surname, I." -- used when the exporter wrote
# the citation with no quotes at all, so commas alone cannot delimit the field
CITE_START = re.compile(r',\s*(?=[A-Z][A-Za-z’\'\-]{1,25},\s*[A-Z]\.)')
CITE_YEAR = re.compile(r'\((?:19|20)\d\d[a-z]?\)')


def split_on_citation(rec):
    """Find where the origin field starts and cut the record there.

    Two delimiters are possible -- `,""` (the exporter quoted the citation) and a
    bare `, Surname, I.` (it did not). Both are ambiguous on their own: an item
    whose own text contains `;"` also produces a `,""`, and the scale/item comma
    looks identical. So a candidate only counts when what follows actually looks
    like an origin (an APA year, or a URL) and what precedes it still holds both
    a scale and an item.
    """
    cands = [(m.start(), m.end()) for m in CITE_SPLIT.finditer(rec)]
    cands += [(m.start(), m.end()) for m in CITE_START.finditer(rec)]
    for start, end in sorted(cands):
        head, tail = rec[:start], rec[end:]
        if "," not in head.lstrip('"'):        # that comma split scale from item
            continue
        probe = tail.lstrip('" ')
        if CITE_YEAR.search(tail[:250]) or probe.startswith("http"):
            return head, tail
    return None, None


def parse_record(rec):
    """-> (scale, item, origin) or None.

    Two record shapes exist and must be split differently:

    A) plain  `scale,item,url,,`  -- the scale may contain UNQUOTED commas
       ("lesbian, gay, bisexual identity"), so anchor from the RIGHT: the last
       four fields are always item, origin, '', ''.
    B) double-wrapped with a citation as origin: `"scale, item,""citation"",,"`.
       Here the citation contains commas and periods, and when the record also
       spanned physical lines the exporter destroyed its quoting -- so split on
       the `,""` that introduces the citation rather than trusting CSV quotes.
    """
    head, tail = split_on_citation(rec)
    if head is not None:
        head = head.lstrip('"').strip()
        scale, _, item = head.partition(",")           # shape B: scale has no comma
        origin = re.sub(r'["\s]*,\s*,\s*"?\s*$', "", tail).strip().strip('"')
        if scale.strip() and item.strip():
            return scale, item, origin

    rows = list(csv.reader(io.StringIO(rec)))
    if not rows:
        return None
    f = rows[0]
    # double-wrapped but well-formed: whole record came back as one field
    if len(f) == 1 and "," in f[0]:
        inner = list(csv.reader(io.StringIO(f[0])))
        if inner:
            f = inner[0]
    # unterminated inner quote leaves 'scale,"item' fused in field 0
    if len(f) == 4 and "," in f[0]:
        head, _, rest = f[0].partition(",")
        f = [head, rest.lstrip('"'), f[1], f[2], f[3]]
    if len(f) < 5:
        return None
    scale = ", ".join(p.strip() for p in f[:-4] if p.strip())
    item, origin = f[-4], f[-3]
    return scale, item, origin


def squish(s):
    s = (s or "").replace("\xa0", " ")
    # nested CSV escaping turns a literal quote inside an item into "" or """"
    s = re.sub(r'"{2,}', '"', s)
    s = re.sub(r"\s+", " ", s).strip()
    if s.count('"') == 1:          # lone wrapping quote left by the mangled export
        s = s.strip('"')
    return s.strip()


def build():
    recs = logical_records(decode(RAW))
    rows, dropped, seen = [], 0, set()
    for rec in recs:
        p = parse_record(rec)
        if not p:
            dropped += 1
            continue
        scale, item, origin = (squish(x) for x in p)
        if (scale.lower(), item.lower()) == ("scale", "item"):   # the export's header row
            continue
        # an item must have real content; scale names are lowercased downstream
        if not scale or len(item) < 3:
            dropped += 1
            continue
        key = (scale.lower(), item.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append((scale.lower(), item, origin))
    return rows, len(recs), dropped


def main():
    rows, n_recs, dropped = build()
    print(f"logical records: {n_recs}; usable items: {len(rows)}; dropped/dupe: {n_recs - len(rows)}")
    n_scales = len(set(r[0] for r in rows))
    print(f"distinct scales: {n_scales}")
    cited = sum(1 for r in rows if not r[2].startswith("http") and r[2])
    print(f"items whose origin is a citation rather than a URL: {cited}")

    if "--check" in sys.argv:
        old = list(csv.DictReader(open(OUT, encoding="utf-8")))
        old_items = set((r["scale"], r["item"]) for r in old)
        new_items = set((r[0], r[1]) for r in rows)
        print(f"\nold file: {len(old)} rows; new: {len(rows)}")
        print(f"  items only in old (corrupt/lost): {len(old_items - new_items)}")
        print(f"  items only in new (recovered/fixed): {len(new_items - old_items)}")
        bad_old = [r for r in old if '""' in r["item"] or "" in r["item"]]
        print(f"  old rows with glued citations or U+0097: {len(bad_old)}")
        bad_new = [r for r in rows if '""' in r[1] or "" in r[1]]
        print(f"  new rows with glued citations or U+0097: {len(bad_new)}")
        return

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "item", "origin"])
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
