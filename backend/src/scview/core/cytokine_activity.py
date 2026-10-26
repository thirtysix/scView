"""Cytokine response, CytoSig-style: which cytokines' signalling a group's expression looks like.

CytoSig (Jiang et al. 2021, Nat Methods, PMID 33972734) built a signature matrix of
43 cytokines x 4,881 genes: each gene's median log fold change over more than 2,000
cytokine-treatment experiments. A profile (a group's expression minus a reference,
per gene) is regressed on the signatures with ridge regression (lambda 10,000) over
the genes both share, both sides scaled to mean 0 and SD 1 per column first (CytoSig's
default normalisation), and each coefficient is turned into a z-score against
coefficients from profiles with the gene labels shuffled (1,000 permutations in the
paper): z = coefficient / standard deviation of the permuted coefficients, p = the
share of permuted coefficients at least as large in absolute value. Implemented here
from that description (the CytoSig code is GPL-3 and is not used). Against CytoSig
0.0.3 (commit ce53cc7, data_significance 1.3, defaults) on Kang STIM vs CTRL per cell
type, 13 types x 43 cytokines: Spearman of the z-scores 0.9997 (per type >= 0.998);
IFN1 tops every type in both.

A positive z says the profile moves the genes the cytokine's treatment moved, the way
it moved them: a response to the signal, not production of the cytokine.

Two kinds of profile, both on log-normalised expression (one solve for all groups):

- ``groups``: each group's mean minus the mean of the rest (or of all cells);
- ``conditions``: within each group, condition A's mean minus condition B's.

The signature file (``signature.centroid``) is fetched once, on first use, into
``DATA_DIR/networks/cytosig`` from a pinned commit of the CytoSig repository and
checked against its sha256; it is GPL-3.0 data, fetched at runtime and never shipped.
``CYTOSIG_AUTO_DOWNLOAD=false`` turns fetching off (a file placed there by hand is
still used when its sha256 matches). Human symbols: a mouse dataset is matched by
upper-casing its symbols (Ifng -> IFNG), which finds most one-to-one orthologues and
misses the genes whose names differ.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import logging
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scview.core import results_store

logger = logging.getLogger(__name__)

COMMIT = "ce53cc7f496f996dc0f759652e21afcd0f20bb8e"
URL = f"https://raw.githubusercontent.com/data2intelligence/CytoSig/{COMMIT}/CytoSig/signature.centroid"
SHA256 = "60a3152afe429a7dbb41fa0597ab9730141ce858af16a35f7063e64bfc239bde"
LICENCE = "GPL-3.0 (data file, fetched at runtime, not redistributed)"
CITATION = "Jiang et al. 2021, Nat Methods (PMID 33972734)"
FILENAME = "signature.centroid"
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024  # the file is 4.3 MB

LAMBDA = 10_000.0  # the paper's ridge penalty
N_PERMUTATIONS = 1000  # the paper's
SEED = 0
MIN_SHARED_GENES = 500
MIN_CELLS = 10  # per side of a profile
MIN_FILE_CYTOKINES, MIN_FILE_GENES = 20, 1000  # a smaller file is not the signature
TAG = "result:cytokines"
KIND = "cytokine"  # in the results store


class SignatureUnavailable(RuntimeError):
    """The signature is not on disk and could not be fetched (offline, off, bad file)."""


# ---------------------------------------------------------------------------
# The signature file


def parse(raw: bytes) -> pd.DataFrame:
    """genes x cytokines; raises ValueError when the bytes are not the signature."""
    try:
        # The header names the cytokines only, so the gene column becomes the index.
        df = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0)
    except Exception as e:  # noqa: BLE001 - any parse failure means a bad file
        raise ValueError(f"not a tab-separated table ({e})") from None
    df.index = df.index.astype(str)
    df.columns = [str(c) for c in df.columns]
    if df.shape[1] < MIN_FILE_CYTOKINES or df.shape[0] < MIN_FILE_GENES:
        raise ValueError(f"expected ~43 cytokines x ~4,900 genes, got {df.shape[1]} x {df.shape[0]}")
    for c in ("IFN1", "IFNG", "TNFA"):
        if c not in df.columns:
            raise ValueError(f"no '{c}' signature")
    values = df.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("non-finite signature values")
    df = df[~df.index.duplicated()]
    return df.astype(np.float64)


def _meta_path(dest: Path) -> Path:
    return dest / "cytosig.json"


def read_meta(dest: Path) -> dict | None:
    if not (dest / FILENAME).is_file():
        return None
    try:
        return json.loads(_meta_path(dest).read_text())
    except (OSError, ValueError):
        return {}


def download(dest: Path, client: httpx.Client | None = None) -> None:
    """Fetch, check (sha256, parse) and store the signature. Raises on any failure,
    leaving no partial file behind."""
    dest.mkdir(parents=True, exist_ok=True)
    part = dest / f"{FILENAME}.part"
    own = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=True)
    try:
        digest, size = hashlib.sha256(), 0
        with client.stream("GET", URL) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes():
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes")
                    digest.update(chunk)
                    fh.write(chunk)
        if digest.hexdigest() != SHA256:
            raise ValueError(f"sha256 {digest.hexdigest()[:12]}... is not the pinned {SHA256[:12]}...")
        sig = parse(part.read_bytes())
        part.replace(dest / FILENAME)
        _meta_path(dest).write_text(json.dumps({
            "name": "CytoSig signature.centroid", "url": URL, "version": f"commit {COMMIT[:12]}",
            "licence": LICENCE, "citation": CITATION, "sha256": SHA256, "download_bytes": size,
            "n_genes": int(sig.shape[0]), "n_cytokines": int(sig.shape[1]),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, indent=1))
        logger.info("Fetched the CytoSig signature: %d genes x %d cytokines", *sig.shape)
    finally:
        part.unlink(missing_ok=True)
        if own:
            client.close()


_lock = threading.Lock()
_loaded: dict[str, tuple[float, pd.DataFrame]] = {}


def load(dest: Path, *, auto_download: bool = True, client: httpx.Client | None = None) -> pd.DataFrame:
    """The signature (genes x cytokines), fetched first when missing. Raises
    ``SignatureUnavailable`` with what to do."""
    path = dest / FILENAME
    with _lock:
        if not path.is_file():
            if not auto_download:
                raise SignatureUnavailable(
                    "The CytoSig signature is not downloaded and CYTOSIG_AUTO_DOWNLOAD is off. "
                    f"Place {FILENAME} (from {URL}) in {dest}, or turn the switch on.")
            try:
                dest.mkdir(parents=True, exist_ok=True)
                with open(dest / ".cytosig.lock", "w") as lk:
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    if not path.is_file():
                        download(dest, client=client)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not fetch the CytoSig signature: %s", e)
                raise SignatureUnavailable(
                    f"The CytoSig signature is not downloaded yet and fetching it failed ({e}). "
                    "scView fetches it once from raw.githubusercontent.com; check the internet "
                    "connection and try again.") from e
        mtime = path.stat().st_mtime
        cached = _loaded.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SHA256:
            raise SignatureUnavailable(
                f"{path} is not the pinned CytoSig signature (sha256 differs); delete it to fetch again.")
        sig = parse(raw)
        _loaded[str(path)] = (mtime, sig)
        return sig


def signature_dir() -> Path:
    from scview.config import get_settings

    return Path(get_settings().DATA_DIR) / "networks" / "cytosig"


# ---------------------------------------------------------------------------
# Matching the dataset's genes


@dataclass
class GeneMatch:
    positions: np.ndarray  # dataset gene positions, aligned with ``signature`` rows
    signature: pd.DataFrame  # the shared genes x cytokines
    mapped_by: str


def match_genes(sig: pd.DataFrame, genes, species: str) -> GeneMatch:
    """The signature genes found in the dataset: exact symbols for human (and unknown),
    upper-cased symbols for mouse (a symbol two dataset genes share is left out)."""
    names = pd.Index(np.asarray(genes, dtype=str))
    if species == "mouse":
        key = names.str.upper()
        mapped_by = "mouse symbols upper-cased to human (orthologues with the same name)"
    else:
        key = names
        mapped_by = "human gene symbols"
    lookup = pd.Series(np.arange(len(key)), index=key)
    lookup = lookup[~lookup.index.duplicated(keep=False)]
    pos = lookup.reindex(sig.index).fillna(-1).to_numpy(np.int64)
    ok = pos >= 0
    return GeneMatch(positions=pos[ok], signature=sig.loc[ok], mapped_by=mapped_by)


# ---------------------------------------------------------------------------
# Group sums -> profiles


def group_sums(M, cols: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(k x len(cols) sums of each code's cells over the genes ``cols``, cells per code)
    of a cells x genes matrix in memory (code -1 = left out)."""
    codes = np.asarray(codes)
    ok = codes >= 0
    sizes = np.bincount(codes[ok], minlength=k).astype(np.int64)
    S = sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))), shape=(k, len(codes)))
    sub = M[:, cols] if sp.issparse(M) else np.asarray(M)[:, cols]
    out = S @ sub
    out = out.toarray() if sp.issparse(out) else np.asarray(out)
    return np.asarray(out, dtype=np.float64), sizes


def group_sums_store(store, cols: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``group_sums`` from a large dataset's gene store (``core/gene_store.py``): one pass
    over the gene-major blocks, only the wanted genes of each."""
    codes = np.asarray(codes)
    ok = codes >= 0
    sizes = np.bincount(codes[ok], minlength=k).astype(np.int64)
    S = sp.csr_matrix((np.ones(int(ok.sum())), (codes[ok], np.flatnonzero(ok))), shape=(k, len(codes)))
    cols = np.asarray(cols, dtype=np.int64)
    out = np.zeros((k, len(cols)), dtype=np.float64)
    order = np.argsort(cols)
    sorted_cols = cols[order]
    for g0, g1, block in store.blocks():
        a, b = np.searchsorted(sorted_cols, g0), np.searchsorted(sorted_cols, g1)
        if a == b:
            continue
        sub = block[:, sorted_cols[a:b] - g0]
        out[:, order[a:b]] = np.asarray((S @ sub).toarray(), dtype=np.float64)
    return out, sizes


# ---------------------------------------------------------------------------
# Ridge regression with a permutation z-score


def standardise(A: np.ndarray) -> np.ndarray:
    """Each column to mean 0 and standard deviation 1 over the genes (a flat column to 0)."""
    A = A - A.mean(axis=0)
    sd = A.std(axis=0, ddof=1)
    return np.divide(A, sd, out=np.zeros_like(A), where=sd > 0)


def ridge_z(X: np.ndarray, Y: np.ndarray, *, lam: float = LAMBDA, n_perm: int = N_PERMUTATIONS,
            seed: int = SEED, chunk: int = 50, normalise: bool = True) -> dict[str, np.ndarray]:
    """Ridge coefficients of every profile (columns of ``Y``, genes x profiles) on the
    signatures (``X``, genes x cytokines), and their permutation z-scores and p-values.
    With ``normalise`` (CytoSig's default), the columns of X and Y are first scaled to
    mean 0 and SD 1 over the shared genes, so lambda means the same on any data.

    beta = (X'X + lam I)^-1 X'Y: one k x genes operator for every profile. Each
    permutation shuffles the gene labels of Y (the same shuffle for all profiles);
    z = beta / sd(permuted beta), p = (#|permuted| >= |beta| + 1) / (n_perm + 1)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if normalise:
        X, Y = standardise(X), standardise(Y)
    g, k = X.shape
    T = np.linalg.solve(X.T @ X + lam * np.eye(k), X.T)  # k x g
    beta = T @ Y
    rng = np.random.default_rng(seed)
    s1 = np.zeros_like(beta)
    s2 = np.zeros_like(beta)
    hits = np.zeros_like(beta)
    absb = np.abs(beta)
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = np.stack([rng.permutation(g) for _ in range(m)])  # m x g
        Yp = Y[perms]  # m x g x s
        B = np.einsum("kg,mgs->mks", T, Yp, optimize=True)
        s1 += B.sum(axis=0)
        s2 += (B * B).sum(axis=0)
        hits += (np.abs(B) >= absb).sum(axis=0)
        done += m
    mean = s1 / n_perm
    sd = np.sqrt(np.maximum(s2 / n_perm - mean**2, 0.0) * n_perm / max(n_perm - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, beta / sd, 0.0)
    p = (hits + 1.0) / (n_perm + 1.0)
    return {"beta": beta, "z": z, "p": p, "sd": sd}


# ---------------------------------------------------------------------------
# One request


@dataclass
class Design:
    """Which cells make each profile: ``codes`` per cell and how to read the sums."""

    mode: str  # "groups" | "conditions"
    groups: list[str]
    codes: np.ndarray
    k: int  # number of codes


def labels_of(col: pd.Series) -> tuple[list[str], np.ndarray]:
    """The groups present (the column's order when categorical) and each cell's index."""
    labels = col.astype(str).where(col.notna())
    present = set(labels.dropna())
    if isinstance(col.dtype, pd.CategoricalDtype):
        order = [str(c) for c in col.cat.categories if str(c) in present]
    else:
        order = sorted(present)
    return order, pd.Index(order).get_indexer(labels)


def design_groups(obs: pd.DataFrame, groupby: str) -> Design:
    groups, codes = labels_of(obs[groupby])
    return Design("groups", groups, codes, len(groups))


def design_conditions(obs: pd.DataFrame, groupby: str | None, condition: str, a: str, b: str) -> Design:
    cond = obs[condition].astype(str).where(obs[condition].notna())
    side = np.where(cond == a, 0, np.where(cond == b, 1, -1))
    if groupby:
        groups, gcodes = labels_of(obs[groupby])
    else:
        groups, gcodes = ["all cells"], np.zeros(len(obs), dtype=np.int64)
    codes = np.where((gcodes >= 0) & (side >= 0), gcodes * 2 + side, -1)
    return Design("conditions", groups, codes, 2 * len(groups))


def profiles(design: Design, sums: np.ndarray, sizes: np.ndarray, reference: str = "rest",
             min_cells: int = MIN_CELLS) -> tuple[np.ndarray, list[int], list[dict], list[list[int]]]:
    """(genes x kept-groups profiles, kept group indices, skipped groups, cells per kept
    group: [n] or [n_a, n_b])."""
    kept, skipped, ns, cols = [], [], [], []
    if design.mode == "groups":
        total, n_all = sums.sum(axis=0), int(sizes.sum())
        for i, g in enumerate(design.groups):
            n = int(sizes[i])
            n_ref = n_all - n if reference == "rest" else n_all
            if n < min_cells or n_ref < min_cells:
                skipped.append({"group": g, "reason": f"fewer than {min_cells} cells on one side"})
                continue
            ref = (total - sums[i]) / n_ref if reference == "rest" else total / n_all
            cols.append(sums[i] / n - ref)
            kept.append(i)
            ns.append([n])
    else:
        for i, g in enumerate(design.groups):
            na, nb = int(sizes[2 * i]), int(sizes[2 * i + 1])
            if na < min_cells or nb < min_cells:
                skipped.append({"group": g, "reason": f"{na} vs {nb} cells (need {min_cells} each)"})
                continue
            cols.append(sums[2 * i] / na - sums[2 * i + 1] / nb)
            kept.append(i)
            ns.append([na, nb])
    Y = np.stack(cols, axis=1) if cols else np.zeros((sums.shape[1], 0))
    return Y, kept, skipped, ns


def score(sig_match: GeneMatch, design: Design, sums: np.ndarray, sizes: np.ndarray, *,
          reference: str = "rest", n_perm: int = N_PERMUTATIONS, min_cells: int | None = None,
          min_genes: int | None = None) -> dict[str, Any]:
    """The whole result for one design: z, p and coefficients per kept group x cytokine."""
    min_cells = MIN_CELLS if min_cells is None else min_cells
    min_genes = MIN_SHARED_GENES if min_genes is None else min_genes
    expressed = sums.sum(axis=0) > 0  # a gene no profiled cell expresses carries nothing
    n_shared = int(expressed.sum())
    if n_shared < min_genes:
        raise ValueError(
            f"Only {n_shared} of the {len(sig_match.signature)} CytoSig genes matched and are "
            f"expressed here; at least {min_genes} are needed for a meaningful fit (the "
            "signature uses human gene symbols).")
    Y, kept, skipped, ns = profiles(design, sums[:, expressed], sizes, reference, min_cells)
    if not kept:
        raise ValueError("No group has enough cells on both sides to make a profile "
                         f"(at least {min_cells} each).")
    X = sig_match.signature.to_numpy()[expressed]
    fit = ridge_z(X, Y, n_perm=n_perm)
    cytokines = list(sig_match.signature.columns)
    groups = [design.groups[i] for i in kept]
    z = fit["z"].T  # groups x cytokines
    top = []
    for gi, g in enumerate(groups):
        order = np.argsort(-z[gi])[:5]
        top.append({"group": g, "cytokines": [
            {"cytokine": cytokines[j], "z": round(float(z[gi, j]), 2), "p": float(fit["p"][j, gi])}
            for j in order]})
    return {
        "cytokines": cytokines, "groups": groups, "group_cells": ns, "skipped": skipped,
        "n_shared_genes": n_shared, "n_signature_genes": int(len(sig_match.signature)),
        "z": np.round(z, 3).tolist(), "p": np.round(fit["p"].T, 5).tolist(),
        "coef": np.round(fit["beta"].T, 6).tolist(), "top": top,
        "lambda": LAMBDA, "n_permutations": int(n_perm), "seed": SEED,
    }


# ---------------------------------------------------------------------------
# Stored results (reuse + the co-pilot)

_results: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_results_lock = threading.Lock()


def recall(adaptor, key: tuple) -> dict | None:
    try:
        with _results_lock:
            r = (_results.get(adaptor) or {}).get("by_key", {}).get(key)
    except TypeError:
        return None
    if r is None:
        stored = results_store.latest(adaptor, KIND, repr(key))
        r = stored["result"] if stored else None
        if r is not None:
            with _results_lock:
                entry = _results.setdefault(adaptor, {"by_key": {}, "last": None})
                entry["by_key"].setdefault(key, r)
    return r


def remember(adaptor, key: tuple, result: dict) -> None:
    """Keep a result as long as the loaded dataset is (a few per dataset; the last one
    is the co-pilot's fact)."""
    try:
        with _results_lock:
            entry = _results.setdefault(adaptor, {"by_key": {}, "last": None})
            new = entry["by_key"].get(key) is not result  # not a reused one shown again
            entry["by_key"][key] = result
            while len(entry["by_key"]) > 8:
                entry["by_key"].pop(next(iter(entry["by_key"])))
            entry["last"] = result
    except TypeError:  # not weak-referenceable (a test double)
        return
    if new:
        results_store.keep(adaptor, KIND, {"key": key, "result": result}, key=repr(key),
                           label=f"Cytokine response: {describe(result)}")


def describe(r: dict) -> str:
    if r["mode"] == "conditions":
        return f"{r['condition']} {r['a']} vs {r['b']}" + (f" within each '{r['groupby']}' group"
                                                          if r.get("groupby") else "")
    return f"each '{r['groupby']}' group vs {'the rest' if r['reference'] == 'rest' else 'all cells'}"


def grounding_lines(adaptor) -> tuple[list[str], str | None]:
    """For the co-pilot: the last cytokine-response run, top cytokines per group."""
    try:
        with _results_lock:
            r = (_results.get(adaptor) or {}).get("last")
    except TypeError:
        return [], None
    if not r:
        stored = results_store.latest(adaptor, KIND)
        r = stored["result"] if stored else None
    if not r:
        return [], None
    detail = (f"CytoSig cytokine response, {describe(r)}: ridge regression on {r['n_shared_genes']} "
              f"shared genes, {r['n_permutations']} permutations")
    lines = ["\n## Cytokine response (CytoSig; signalling response, not cytokine production)",
             f"- {detail} [{TAG}]."]
    for t in r["top"][:12]:
        sig = [c for c in t["cytokines"] if c["z"] >= 2][:3]
        if not sig:
            lines.append(f"- cytokine response (CytoSig): {t['group']}: no cytokine with z >= 2")
            continue
        what = ", ".join(f"{c['cytokine']} z={c['z']:.1f}" for c in sig)
        suffix = f" {r['a']} vs {r['b']}" if r["mode"] == "conditions" else ""
        lines.append(f"- cytokine response (CytoSig): {t['group']}{suffix} top {what}")
    if r["species"] == "mouse":
        lines.append("- Caveat: human signature; mouse genes matched by upper-cased symbol.")
    return lines, detail
