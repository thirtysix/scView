"""Cell Ontology (CL) terms for cell labels.

The ontology is the ``cl-basic.obo`` that ``core/marker_db.py`` fetches on first use into
``DATA_DIR/marker_db/raw`` (``MARKER_DB_AUTO_DOWNLOAD=false`` turns fetching off; a file
put there by hand is used as it is). Only that one file is fetched for a search, not the
marker databases. It is parsed once into ``Index`` (names, synonyms, definitions, is_a
parents) and searched in memory.

A label column's terms are kept in ``uns[TERMS_KEY]`` as JSON, ``{column: {label: id}}``
(JSON because a label may hold a '/', which h5py reads as a path). Every id in it was
chosen or confirmed by the user: suggestions (``suggest``, ``marker_names.Matcher``) are
only shown, never stored. ``UNKNOWN`` is allowed as a term, the value CELLxGENE takes for
cells of no known type. A renamed label keeps its term; a label merged into another
takes the other's term. Entries of labels that left a column are kept, so an undone
labelling brings its term back; only present labels are reported.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from scview.core import marker_db as mdb
from scview.core.marker_names import HOW_TEXT, Matcher, Ontology, norm, toks

logger = logging.getLogger(__name__)

TERMS_KEY = "scview_label_terms"
TAG = "result:ontology"
UNKNOWN = "unknown"
OLS_URL = "https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id={}"
MAX_RESULTS = 50
N_SUGGEST = 3
_ID = re.compile(r"^CL[:_](\d{7})$", re.I)
# Synonym scopes searched: BROAD ones ("suppressor T cell" for Treg) name a wider group.
_SCOPES = ("EXACT", "NARROW", "RELATED")
# Labels that name no cell type: nothing is suggested for them.
_NO_TYPE = {"unassigned", "unknown", "nan", "none", "na", "n/a", "other", "doublet", "doublets"}


class OntologyUnavailable(RuntimeError):
    """The Cell Ontology is not on disk and cannot be fetched now (a 503)."""


class TermError(ValueError):
    """A term request that cannot be served (``status`` is the HTTP code)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# The ontology


@dataclass
class Term:
    id: str
    name: str
    synonyms: list[tuple[str, str]]  # (text, scope)
    definition: str | None
    parents: list[str]


def parse(text: str) -> dict[str, Term]:
    """The non-obsolete CL terms of an OBO file, with definitions and every synonym."""
    terms: dict[str, Term] = {}
    cur: dict | None = None

    def flush(c):
        if c and c.get("id", "").startswith("CL:") and "name" in c and not c.get("obs"):
            terms[c["id"]] = Term(c["id"], c["name"], c["syn"], c.get("def"), sorted(c["is_a"]))

    for line in text.splitlines():
        if line in ("[Term]", "[Typedef]"):
            flush(cur)
            cur = {"is_a": set(), "syn": []} if line == "[Term]" else None
        elif cur is not None and ": " in line:
            k, v = line.split(": ", 1)
            if k in ("id", "name"):
                cur[k] = v.strip()
            elif k == "def" and v.count('"') >= 2:
                cur["def"] = v[1:].rsplit('"', 1)[0].replace('\\"', '"')
            elif k == "is_a":
                p = v.split(" ! ")[0].split(" {")[0].strip()
                if p.startswith("CL:"):
                    cur["is_a"].add(p)
            elif k == "synonym" and v.count('"') >= 2:
                text_, rest = v[1:].split('"', 1)
                scope = rest.split()[0] if rest.split() else ""
                if scope in _SCOPES:
                    cur["syn"].append((text_, scope))
            elif k == "is_obsolete" and v.strip() == "true":
                cur["obs"] = True
    flush(cur)
    return terms


def _flat(s: str) -> str:
    return " ".join(re.sub(r"[-_,/]+", " ", s.lower()).split())


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _signs(s: str) -> str:
    """A trailing minus spelled out ("CD16- monocyte" -> "CD16 negative monocyte"), so
    the name normalisation, which drops punctuation, keeps it apart from "CD16 monocyte"."""
    return re.sub(r"(?<=[A-Za-z0-9])-(?=[\s,/)]|$)", " negative", s)


def _key(s: str, abbr: bool = False) -> frozenset[str]:
    return norm(_signs(s), abbr)


def clean_id(value: str) -> str | None:
    """'CL:0000236' (also 'cl_0000236') in the canonical form, else None."""
    m = _ID.match(value.strip()) if isinstance(value, str) else None
    return f"CL:{m.group(1)}" if m else None


_KIND_RANK = {"name": 0, "synonym": 1, "related": 2}
SUPERSET_EXTRA = 4  # a suggested narrower term adds at most this many words to the label's
# Words CL adds to the usual T cell names that labels leave out ("CD8 T" is the
# "CD8-positive, alpha-beta T cell"): not counted as extra words when ranking.
QUALIFIERS = frozenset({"alpha", "beta", "thymus", "derived"})


@dataclass
class Index:
    """Every term's searchable strings, built once."""

    terms: dict[str, Term]
    version: str | None = None
    _entries: list[tuple[str, str, str, str, frozenset]] = field(default_factory=list)  # id, text, kind, flat, key
    _flat: dict[str, list[int]] = field(default_factory=dict)
    _key: dict[frozenset, list[int]] = field(default_factory=dict)
    _vocab: dict[str, set[int]] = field(default_factory=dict)  # word -> entries
    _keyw: dict[str, set[int]] = field(default_factory=dict)  # normalised word -> entries
    _onto: Ontology | None = None
    _match: Matcher | None = None
    _names: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        for cid, t in self.terms.items():
            strings = [(t.name, "name")] + [(s, "synonym" if sc == "EXACT" else "related") for s, sc in t.synonyms]
            for text, kind in strings:
                i = len(self._entries)
                f, k = _flat(text), _key(text)
                self._entries.append((cid, text, kind, f, k))
                self._flat.setdefault(f, []).append(i)
                if k:
                    self._key.setdefault(k, []).append(i)
                for w in _words(text):
                    self._vocab.setdefault(w, set()).add(i)
                for w in k:
                    self._keyw.setdefault(w, set()).add(i)

    def __len__(self) -> int:
        return len(self.terms)

    def describe(self, cid: str, matched: str | None = None, match: str | None = None) -> dict[str, Any]:
        t = self.terms[cid]
        return {
            "id": cid, "name": t.name, "matched": matched, "match": match,
            "definition": t.definition,
            "parents": [{"id": p, "name": self.terms[p].name} for p in t.parents if p in self.terms],
            "url": OLS_URL.format(cid),
        }

    def name(self, cid: str) -> str | None:
        t = self.terms.get(cid)
        return t.name if t else None

    def search(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        """Terms for ``q``, best first: the id itself; the same name, exact synonym or
        other synonym; the same words (stop words, plurals and the abbreviation table
        aside: "CD14 Mono", "NK"); a name starting with ``q``; every word of ``q``
        starting a word of the term. Within each: whole words first, then fewest other
        words (``QUALIFIERS`` aside), then the more general term."""
        q = (q or "").strip()
        if not q:
            return []
        cid = clean_id(q)
        if cid:
            return [self.describe(cid, None, "id")] if cid in self.terms else []
        best: dict[str, tuple] = {}
        qw = _words(q)
        qk = _key(q, abbr=True)
        onto = self.ontology()

        def hit(i: int, tier: int) -> None:
            # tier: 0 same text, 1 same words, 2 starts with q, 3 every word of q starts a
            # word; then whole words matched, other words, name before synonyms, length
            cid_, _text, kind, f, k = self._entries[i]
            ew = f.split()
            whole = sum(w in ew for w in qw)
            extra = len(k - qk - QUALIFIERS)
            rank = (tier, -whole, extra, -onto.n_desc(cid_), _KIND_RANK[kind], len(f),
                    self.terms[cid_].name.lower(), i)
            if cid_ not in best or rank < best[cid_]:
                best[cid_] = rank

        qf = _flat(q)
        for i in self._flat.get(qf, ()):
            hit(i, 0)
        for k in {_key(q), _key(q, abbr=True)}:
            for i in self._key.get(k, ()) if k else ():
                hit(i, 1)
        if qw:
            cand: set[int] | None = None
            for w in qw:
                found: set[int] = set()
                for v, ids in self._vocab.items():
                    if v.startswith(w):
                        found |= ids
                cand = found if cand is None else cand & found
                if not cand:
                    break
            for i in cand or ():
                hit(i, 2 if self._entries[i][3].startswith(qf) else 3)
        order = sorted(best.items(), key=lambda kv: kv[1])[: max(1, min(limit, MAX_RESULTS))]
        out = []
        for cid_, rank in order:
            _c, text, kind, _f, _k = self._entries[rank[-1]]
            match = (("name" if kind == "name" else "synonym") if rank[0] == 0
                     else ("words", "prefix", "tokens")[rank[0] - 1])
            out.append(self.describe(cid_, text if kind != "name" else None, match))
        return out

    # -- suggestions -------------------------------------------------------

    def ontology(self) -> Ontology:
        if self._onto is None:
            self._onto = Ontology({cid: (t.name, [s for s, sc in t.synonyms if sc == "EXACT"], t.parents)
                                   for cid, t in self.terms.items()})
        return self._onto

    def _matcher(self) -> Matcher:
        """``marker_names.Matcher`` over every term's name and exact synonyms."""
        if self._match is None:
            names: dict[str, str] = {}
            for cid, t in self.terms.items():
                for nm in (t.name, *(s for s, sc in t.synonyms if sc == "EXACT")):
                    names.setdefault(_signs(nm), cid)
            self._names = names
            self._match = Matcher(names, self.ontology())
        return self._match

    def _same(self, key: frozenset) -> list[str]:
        """Terms whose name or exact synonym has these normalised words."""
        out: list[str] = []
        for i in sorted(self._key.get(key, ()), key=lambda i: (_KIND_RANK[self._entries[i][2]], i)):
            cid, _t, kind, _f, _k = self._entries[i]
            if kind != "related" and cid not in out:
                out.append(cid)
        return out

    def _narrower(self, key: frozenset) -> list[str]:
        """Terms holding every word of the label and at most ``SUPERSET_EXTRA`` more, fewest
        extra words first, then the more general term (more descendants)."""
        cand: set[int] | None = None
        for w in key:
            cand = set(self._keyw.get(w, ())) if cand is None else cand & self._keyw.get(w, set())
            if not cand:
                return []
        onto = self.ontology()
        best: dict[str, tuple] = {}
        for i in cand or ():
            cid, _t, kind, f, k = self._entries[i]
            extra = len(k - key - QUALIFIERS)
            if kind == "related" or k == key or extra > SUPERSET_EXTRA:
                continue
            rank = (extra, -onto.n_desc(cid), _KIND_RANK[kind], len(f), cid)
            if cid not in best or rank < best[cid]:
                best[cid] = rank
        return sorted(best, key=best.__getitem__)

    def suggest(self, label: str, n: int = N_SUGGEST) -> list[dict[str, Any]]:
        """Up to ``n`` terms for a free-text label, best first, each with how it matched:
        the same words (with the abbreviation table: "CD14 Mono", "Mk", "pDC"), else
        terms holding every word of the label ("CD4 Naive T" -> naive thymus-derived
        CD4-positive, alpha-beta T cell), then ``marker_names.Matcher``'s nearest names."""
        if not isinstance(label, str) or _flat(label) in _NO_TYPE or not toks(label):
            return []
        own = clean_id(label)
        if own:
            return [{**self.describe(own), "how": "id", "how_text": "the label is the id"}] if own in self.terms else []
        picked: list[tuple[str, str]] = []

        def add(ids, how):
            for c in ids:
                if len(picked) < n and c not in {p for p, _ in picked}:
                    picked.append((c, how))

        for k in (_key(label), _key(label, abbr=True)):
            if k:
                add(self._same(k), "exact")
        if not picked:
            k = _key(label, abbr=True)
            if k:
                add(self._narrower(k), "narrower")
            hit, how = self._matcher().match(_signs(label))
            if how in ("subset", "cl-ancestor"):
                add([self._names[h] for h in hit if h in self._names], how)
        text = {**HOW_TEXT, "narrower": "term holding every word of the label"}
        return [{**self.describe(c), "how": how, "how_text": text.get(how, how)} for c, how in picked]


_cache: dict[str, tuple[tuple, Index]] = {}
_cache_lock = threading.Lock()


def _files(dest: Path) -> tuple[Path, Path]:
    return dest / "raw" / mdb.ONTOLOGY_FILE, mdb._ontology_table(dest)


def _version(dest: Path, text: str | None) -> str | None:
    v = mdb.read_manifest(dest).get("sources", {}).get("ontology", {}).get("version")
    if v:
        return v
    m = re.search(r"^data-version: (.+)$", (text or "")[:5000], re.M)
    return m.group(1).strip() if m else None


def loaded(dest: Path | None = None) -> Index | None:
    """The index from what is on disk (no fetching), or None."""
    dest = dest or mdb.marker_db_dir()
    raw, table = _files(dest)
    src = raw if raw.is_file() else table if table.is_file() else None
    if src is None:
        return None
    stamp = (str(src), src.stat().st_mtime_ns, src.stat().st_size)
    with _cache_lock:
        hit = _cache.get(str(dest))
        if hit and hit[0] == stamp:
            return hit[1]
        text = None
        if src == raw:
            text = raw.read_text(encoding="utf-8", errors="replace")
            terms = parse(text)
        else:  # the compact table only: no definitions
            import gzip

            with gzip.open(table, "rt", encoding="utf-8") as f:
                terms = {k: Term(k, v[0], [(s, "EXACT") for s in v[1]], None, list(v[2]))
                         for k, v in json.load(f).items()}
        if not terms:
            return None
        idx = Index(terms, _version(dest, text))
        idx.ontology().n_desc("CL:0000000")  # descendant counts (ranking) built now, not on a search
        _cache[str(dest)] = (stamp, idx)
        return idx


def cached() -> Index | None:
    """The index if it was built already in this process (never reads or fetches)."""
    hit = _cache.get(str(mdb.marker_db_dir()))
    return hit[1] if hit else None


def get_index(dest: Path | None = None, *, auto_download: bool | None = None, retry: bool = False,
              client=None) -> Index:
    """The index, fetching ``cl-basic.obo`` first when missing and allowed (call in a
    worker thread). Raises ``OntologyUnavailable`` saying why."""
    dest = dest or mdb.marker_db_dir()

    def from_disk() -> Index | None:
        try:
            return loaded(dest)
        except (OSError, ValueError) as e:
            logger.warning("Cell Ontology unreadable: %s", e)
            return None

    idx = from_disk()
    if idx is not None:
        return idx
    st = mdb.ensure(dest, auto_download=auto_download, retry=retry, client=client, parts=("ontology",))
    idx = from_disk()
    if idx is not None:
        return idx
    if not st["auto_download"]:
        raise OntologyUnavailable(
            "The Cell Ontology is not downloaded and MARKER_DB_AUTO_DOWNLOAD is off. "
            f"Put {mdb.ONTOLOGY_FILE} in {dest / 'raw'} or turn the download on.")
    err = st.get("ontology", {}).get("error")
    raise OntologyUnavailable(
        "The Cell Ontology could not be downloaded" + (f" ({err})" if err else "")
        + ". Check the internet connection and try again.")


# ---------------------------------------------------------------------------
# A column's label -> term mapping


def read_all(adata) -> dict[str, dict[str, str]]:
    raw = adata.uns.get(TERMS_KEY)
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        return {}
    return {str(c): {str(k): str(v) for k, v in m.items() if isinstance(v, str)}
            for c, m in data.items() if isinstance(m, dict)}


def _write_all(adata, data: dict[str, dict[str, str]]) -> None:
    data = {c: m for c, m in data.items() if m}
    if data:
        adata.uns[TERMS_KEY] = json.dumps(data, sort_keys=True)  # replaced, never mutated
    else:
        adata.uns.pop(TERMS_KEY, None)


def stored(adata, column: str) -> dict[str, str]:
    """The column's stored label -> id entries (labels that left the column too)."""
    return read_all(adata).get(column, {})


def clean_term(value: object, index: Index | None = None) -> str | None:
    """A term id to store: None (clear), UNKNOWN, or a CL id (in the ontology, when it is
    loaded). TermError otherwise."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == UNKNOWN:
        return UNKNOWN
    cid = clean_id(value) if isinstance(value, str) else None
    if cid is None:
        raise TermError(f"Not a Cell Ontology id: {str(value)[:40]!r} (expected CL:0000000 or 'unknown').")
    if index is not None and cid not in index.terms:
        raise TermError(f"{cid} is not a current Cell Ontology term.")
    return cid


def labels_of(adata, column: str) -> tuple[list[str], dict[str, int], int]:
    """(labels in category order, cells per label, cells with no label)."""
    if column not in adata.obs.columns:
        raise TermError(f"Column '{column}' not found.", status=404)
    s = adata.obs[column]
    if pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype):
        raise TermError(f"'{column}' is not a label column.")
    if not isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype("category")
    counts = s.value_counts(sort=False)
    labels = [str(c) for c in s.cat.categories if counts.get(c, 0) > 0]
    return labels, {str(k): int(v) for k, v in counts.items() if v > 0}, int(s.isna().sum())


def set_terms(adata, column: str, changes: dict[str, Any], index: Index | None = None) -> dict[str, str | None]:
    """Set (an id) or clear (None) the terms of labels of ``column``; everything is
    checked before anything changes. Returns each changed label's previous term."""
    labels, _, _ = labels_of(adata, column)
    present = set(labels)
    clean: dict[str, str | None] = {}
    for label, value in changes.items():
        if label not in present:
            raise TermError(f"'{label}' is not a label of '{column}'.")
        clean[label] = clean_term(value, index)
    data = read_all(adata)
    m = dict(data.get(column, {}))
    previous: dict[str, str | None] = {}
    for label, cid in clean.items():
        if m.get(label) == cid:
            continue
        previous[label] = m.get(label)
        if cid is None:
            m.pop(label, None)
        else:
            m[label] = cid
    if previous:
        data[column] = m
        _write_all(adata, data)
    return previous


def restore_terms(adata, column: str, terms: dict[str, str | None]) -> None:
    """Put back terms a labelling replaced (its ``previous_terms``), present or not."""
    if not terms:
        return
    data = read_all(adata)
    m = dict(data.get(column, {}))
    for label, cid in terms.items():
        clean = clean_term(cid)
        if clean is None:
            m.pop(label, None)
        else:
            m[label] = clean
    data[column] = m
    _write_all(adata, data)


def follow_rename(adata, column: str, old: str, new: str, merge: bool) -> None:
    """A renamed label keeps its term; one merged into another label takes that label's
    term (its own is dropped: it named other cells)."""
    data = read_all(adata)
    m = data.get(column)
    if not m or old not in m:
        return
    m = dict(m)
    cid = m.pop(old)
    if not merge:
        m[new] = cid
    data[column] = m
    _write_all(adata, data)


def drop_columns(adata, columns) -> None:
    """Forget the terms of columns that were removed (or are made again from scratch)."""
    data = read_all(adata)
    gone = [c for c in columns if c in data]
    for c in gone:
        del data[c]
    if gone:
        _write_all(adata, data)


def drop_column(adata, column: str) -> None:
    drop_columns(adata, [column])


def summary(adata, column: str, index: Index | None = None) -> dict[str, Any]:
    """Each present label with its cells and term; whether every label has one."""
    labels, counts, n_missing = labels_of(adata, column)
    m = stored(adata, column)
    rows = []
    for label in labels:
        cid = m.get(label)
        rows.append({
            "label": label, "n_cells": counts.get(label, 0), "id": cid,
            "name": ("unknown cell type" if cid == UNKNOWN
                     else index.name(cid) if (cid and index is not None) else None),
            "url": OLS_URL.format(cid) if cid and cid != UNKNOWN else None,
        })
    missing = [r["label"] for r in rows if not r["id"]]
    return {
        "column": column, "labels": rows, "n_labels": len(rows), "n_mapped": len(rows) - len(missing),
        "missing": missing, "n_cells_unlabelled": n_missing,
        "complete": bool(rows) and not missing and n_missing == 0,
        "version": index.version if index is not None else None,
    }


def mapped_columns(adata) -> list[str]:
    """Columns with at least one stored term that are still in obs."""
    return [c for c, m in read_all(adata).items() if m and c in adata.obs.columns]


def term_ids(adata, column: str) -> pd.Series | None:
    """``cell_type_ontology_term_id`` per cell from the column's mapping, or None unless
    every cell's label has a user-confirmed term."""
    s = summary(adata, column)
    if not s["complete"]:
        return None
    m = stored(adata, column)
    col = adata.obs[column].astype(str)
    out = col.map(m)
    if out.isna().any():  # pragma: no cover - summary said complete
        return None
    return pd.Categorical(out, categories=sorted(set(out)))


def suggest(adata, column: str, index: Index) -> dict[str, Any]:
    labels, counts, _ = labels_of(adata, column)
    m = stored(adata, column)
    rows = []
    for label in labels:
        rows.append({"label": label, "n_cells": counts.get(label, 0), "current": m.get(label),
                     "suggestions": index.suggest(label)})
    return {"column": column, "labels": rows, "n_labels": len(rows),
            "n_suggested": sum(bool(r["suggestions"]) for r in rows), "version": index.version}


# ---------------------------------------------------------------------------
# Co-pilot


def grounding_lines(adata, highlighted: dict | None) -> tuple[list[str], str | None]:
    """For the co-pilot: the term the user gave the highlighted label, if any."""
    if not isinstance(highlighted, dict):
        return [], None
    column, value = highlighted.get("column"), highlighted.get("value")
    if not column or value is None:
        return [], None
    cid = stored(adata, str(column)).get(str(value))
    if not cid:
        return [], None
    try:
        idx = cached() or loaded()  # from disk at most, never fetched here
    except (OSError, ValueError):
        idx = None
    name ="unknown cell type" if cid == UNKNOWN else (idx.name(cid) if idx else None)
    tag = f"{TAG}:{column}"
    what = f"{cid}" + (f" ({name})" if name and cid != UNKNOWN else " (no known cell type)" if cid == UNKNOWN else "")
    line = (f"- Cell Ontology term the user gave '{value}' in '{column}': {what}; chosen by the user, "
            f"not inferred [{tag}]")
    return [f"\n## Cell Ontology term of the highlighted label [{tag}]", line], f"{column}={value}: {cid}"
