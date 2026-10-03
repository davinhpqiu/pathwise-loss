"""Checks for response-metric separation claims.

Two claims about
``d_r(X, Xt) = sup_{||A|| <= r} ||Phi^A(X) - Phi^A(Xt)||``,
where ``Phi^A`` is terminal solution operator of ``dY = A(Y) dX`` and ``A`` is a
tuple of ``n x n`` matrices.

Claim 1, homogeneous blindness.  A Lie identity of ``gl(n)`` is invisible to
every degree-wise response in dimension ``n``.  Full column rank of sampled
Lyndon evaluations modulo a prime certifies absence of a real identity at that
degree.  Rank deficiency only suggests an identity and needs symbolic proof.
Full path blindness additionally needs every relevant degree to vanish; an
arbitrary matrix evaluation does not descend to a finite nilpotent quotient.
Known values anchor the computation:
minimal degree of a Lie identity of ``sl(2)`` is 5 (Kanel-Belov, Malev and Rowen,
arXiv:1506.06792), and ``sl(n)`` satisfies one of degree ``n^2 + 1``.

Claim 2, lower bound attainment.  For a word ``w`` of length ``k`` set
``A_i = k^{-1/2} sum_{j : w_{k+1-j} = i} E_{j+1,j}``.  The response tuple then
has Frobenius norm one, and ``(A_{v_1} ... A_{v_k})_{k+1,1}`` is ``k^{-k/2}``
for ``v = w`` and zero otherwise.  Thus response dimension ``n = k + 1``
isolates one signature coordinate and gives the level-``k`` lower bound
``d_r >= r^k k^{-k/2} sup_w |S^w(X) - S^w(Xt)|``.

Ranks are computed over ``F_p`` rather than in floating point: Lie brackets of
matrices are traceless, so column rank is capped by ``samples * (n^2 - 1)``, and
floating point SVD returned ranks above that cap.

Run::

    python scripts/check_response_separation.py
    python scripts/check_response_separation.py --kmax 9
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

P = (1 << 31) - 1


def lyndon_words(d: int, k: int) -> list[tuple[int, ...]]:
    """Lyndon words of length exactly ``k`` over alphabet ``0..d-1`` (Duval)."""
    out: list[tuple[int, ...]] = []
    w = [-1]
    while w:
        w[-1] += 1
        m = len(w)
        if m == k:
            out.append(tuple(w))
        while len(w) < k:
            w.append(w[-m])
        while w and w[-1] == d - 1:
            w.pop()
    return [t for t in out if len(t) == k]


def is_lyndon(w: tuple[int, ...]) -> bool:
    return all(w < w[i:] + w[:i] for i in range(1, len(w)))


def standard_bracket(w: tuple[int, ...]):
    """Lyndon word to nested bracket via Chen-Fox-Lyndon standard factorisation."""
    if len(w) == 1:
        return w[0]
    for i in range(1, len(w)):
        if is_lyndon(w[i:]):
            return (standard_bracket(w[:i]), standard_bracket(w[i:]))
    raise RuntimeError(f"no standard factorisation for {w}")


def matmul_mod_p(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Matrix product modulo P without overflowing int64 dot-product sums."""
    left, right = (
        np.asarray(left, dtype=np.int64) % P,
        np.asarray(right, dtype=np.int64) % P,
    )
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    for index in range(left.shape[1]):
        # One product plus one reduced residue is < P**2 + P < 2**63.
        result = (result + left[:, index, None] * right[index, None, :]) % P
    return result


def evaluate(tree, A):
    if isinstance(tree, int):
        return A[tree]
    left, right = evaluate(tree[0], A), evaluate(tree[1], A)
    return (matmul_mod_p(left, right) - matmul_mod_p(right, left)) % P


def rank_mod_p(M: np.ndarray) -> int:
    """Gaussian elimination modulo ``P``; entries fit int64 since P < 2^31."""
    M = M.copy() % P
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), P - 2, P)) % P
        col = M[:, c].copy()
        col[r] = 0
        hit = np.nonzero(col)[0]
        if hit.size:
            M[hit] = (M[hit] - np.outer(col[hit], M[r])) % P
        r += 1
        if r == rows:
            break
    return r


def identity_table(d: int, n: int, kmax: int, seed: int = 1):
    """Rank deficiency of degree-k Lie evaluation into gl(n), k = 2..kmax."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(2, kmax + 1):
        words = lyndon_words(d, k)
        if not words:
            continue
        trees = [standard_bracket(w) for w in words]
        dim = len(words)
        samples = max(8, dim // max(1, n * n - 1) + 8)
        blocks = []
        for _ in range(samples):
            A = [rng.integers(0, P, size=(n, n), dtype=np.int64) for _ in range(d)]
            blocks.append(
                np.array([evaluate(t, A).ravel() for t in trees], dtype=np.int64)
            )
        rank = rank_mod_p(np.hstack(blocks))
        out.append((k, dim, rank, dim - rank, samples * (n * n - 1)))
    return out


def isolation_check(dmax: int = 3, kmax: int = 5) -> tuple[int, int]:
    """Verify single-word isolation by nilpotent shifts. Returns (checked, failures)."""
    checked = failures = 0
    for d in range(2, dmax + 1):
        for k in range(1, kmax + 1):
            for w in product(range(d), repeat=k):
                n = k + 1
                A = [np.zeros((n, n)) for _ in range(d)]
                for j, letter in enumerate(reversed(w)):
                    A[letter][j + 1, j] = 1.0
                for a in A:
                    if a.any() and abs(np.linalg.norm(a, 2) - 1) > 1e-12:
                        failures += 1
                for v in product(range(d), repeat=k):
                    M = np.eye(n)
                    for letter in v:
                        M = M @ A[letter]
                    checked += 1
                    if abs(M[k, 0] - (1.0 if v == w else 0.0)) > 1e-12:
                        failures += 1
    return checked, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, help="new directory for ranks and provenance")
    args = ap.parse_args()
    if args.kmax < 2:
        ap.error("--kmax must be at least two")
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=False)

    print(
        "Sampled Lie evaluation ranks; deficient ranks require symbolic verification\n"
    )
    records = []
    for d in (2, 3, 4):
        for n in (2, 3, 4):
            kmax = args.kmax if d == 2 else min(args.kmax, 8 if d == 3 else 6)
            print(f"d={d} channels, n={n} response dimension")
            print("   k   dim L_k    rank   deficiency   column cap")
            for k, dim, rank, defi, cap in identity_table(d, n, kmax, args.seed):
                mark = "   <-- candidate identity" if defi else ""
                print(f"  {k:2d} {dim:8d} {rank:7d} {defi:11d} {cap:12d}{mark}")
                records.append(
                    {
                        "driver_dim": d,
                        "response_dim": n,
                        "level": k,
                        "lie_dim": dim,
                        "rank": rank,
                        "deficiency": defi,
                        "trace_rank_cap": cap,
                        "status": "candidate_identity" if defi else "no_real_identity",
                    }
                )
            print()

    checked, failures = isolation_check()
    print(f"Word isolation: {checked} word pairs checked, {failures} failures")
    if args.out is not None:
        with (args.out / "ranks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        source = Path(__file__).resolve()
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "prime": P,
            "seed": args.seed,
            "kmax": args.kmax,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "source_sha256": {
                "scripts/check_response_separation.py": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            },
            "isolation_checked": checked,
            "isolation_failures": failures,
            "interpretation": "Full rank certifies absence; deficiency is a candidate only.",
        }
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise RuntimeError("word-isolation verification failed")


if __name__ == "__main__":
    main()
