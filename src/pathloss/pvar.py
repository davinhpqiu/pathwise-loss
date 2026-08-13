r"""p-variation of a sampled path.

For a sampled path :math:`x_0, \dots, x_N` in a metric space :math:`(E, d)` and
:math:`p \ge 1`,

.. math::
    V_p(x)^p \;=\; \sup \sum_k d(x_{n_k}, x_{n_{k-1}})^p

over increasing subsequences :math:`0 = n_0 < n_1 < \dots < n_K = N`.

Three implementations of that supremum, in increasing order of speed and
decreasing order of obviousness:

===========================  ==================  ==========================
function                     cost                use
===========================  ==================  ==========================
``p_variation_brute``        :math:`2^{N-1}`     oracle for tests, N <~ 20
``p_variation_exact``        :math:`O(N^2)`      reference implementation
``p_variation_pruned``       :math:`O(N\log N)`  production; worst case N^2
===========================  ==================  ==========================

All three take the supremum over subsequences of the **observed** grid, so all
three are lower bounds on the p-variation of the underlying continuous path.
No finite sample determines that quantity.

"Dyadic" appears in ``p_variation_pruned`` only as the shape of a search tree.
It changes no answer. An earlier ``p_variation_dyadic``, which restricted the
partitions considered and so returned a weaker lower bound, was removed on
13/08: a second, worse estimator of the same quantity earns nothing.

Convention: these return the p-th root, unlike the reference C++ below.

References
----------
Korepanov, A., Lyons, T., and Zorin-Kranich, P. `p-var
<https://github.com/khumarahn/p-var>`_. ``p_variation_pruned`` is a port of
``p_var_backbone`` from ``p_var.h``, written from the published source.
Butkus, V., and Norvaisa, R. (2018), *Computation of p-variation*, Lith. Math.
J. 58(4). A faster algorithm specific to real-valued paths, not implemented.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "p_variation_brute",
    "p_variation_exact",
    "p_variation_pruned",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_points(x: np.ndarray) -> np.ndarray:
    """(T,) or (T, d) -> (T, d) float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must have shape (T,) or (T, d)")
    if x.shape[0] < 1:
        raise ValueError("need at least one point")
    return x


def _index_metric(x: np.ndarray, dist):
    """Turn a point metric into one taking indices into x."""
    if dist is None:
        def d(a: int, b: int) -> float:
            return float(np.sqrt(np.sum((x[a] - x[b]) ** 2)))
    else:
        def d(a: int, b: int) -> float:
            return float(dist(x[a], x[b]))
    return d


def _check_p(p: float) -> float:
    p = float(p)
    if p < 1.0:
        raise ValueError("p must be >= 1")
    return p


# ---------------------------------------------------------------------------
# brute force: the oracle
# ---------------------------------------------------------------------------

def p_variation_brute(x, p: float = 2.0, dist=None, max_points: int = 20) -> float:
    r"""Supremum by enumeration of all :math:`2^{N-1}` subsequences.

    Exists to test the other two. Correct by construction: it contains no
    argument that could be wrong, only a loop over every candidate. Refuses
    inputs large enough to be slow, since a slow test does not get run.

    Parameters
    ----------
    x : (T,) or (T, d) array of points.
    p : exponent, p >= 1.
    dist : optional metric on points. Euclidean if omitted.
    max_points : refuse inputs longer than this.
    """
    p = _check_p(p)
    x = _as_points(x)
    n = x.shape[0]
    if n == 1:
        return 0.0
    if n > max_points:
        raise ValueError(
            f"{n} points would need 2^{n - 2} subsets; "
            f"raise max_points deliberately if that is intended"
        )
    d = _index_metric(x, dist)

    interior = n - 2                       # points 1 .. n-2 are optional
    best = 0.0
    for mask in range(1 << interior):
        idx = [0]
        for i in range(interior):
            if (mask >> i) & 1:
                idx.append(i + 1)
        idx.append(n - 1)
        total = 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            total += d(a, b) ** p
        if total > best:
            best = total
    return float(best ** (1.0 / p))


# ---------------------------------------------------------------------------
# exact: the O(N^2) dynamic programme
# ---------------------------------------------------------------------------

def p_variation_exact(x, p: float = 2.0, dist=None, return_points: bool = False):
    r"""Supremum over subsequences of the observed grid, by dynamic programming.

    Any subsequence ending at :math:`j` has a last step, from some :math:`m`.
    Given :math:`m`, everything before it must itself be optimal for the prefix
    ending at :math:`m`, since substituting a better prefix leaves the last step
    untouched. Hence

    .. math::
        D[j] = \max_{m<j}\big(D[m] + d(x_m, x_j)^p\big), \qquad D[0] = 0,

    and :math:`V_p = D[N]^{1/p}`. Cost :math:`O(N^2)` in time, :math:`O(N)` in
    memory beyond the path itself.

    Parameters
    ----------
    x : (T,) or (T, d) array of points.
    p : exponent, p >= 1.
    dist : optional metric on points. Euclidean if omitted, which is also the
        vectorised path; a supplied metric costs one Python call per pair.
    return_points : also return the maximising subsequence.

    Returns
    -------
    float, or (float, list[int]) if `return_points`.
    """
    p = _check_p(p)
    x = _as_points(x)
    n = x.shape[0]
    if n == 1:
        return (0.0, [0]) if return_points else 0.0

    best = np.zeros(n)
    link = np.zeros(n, dtype=int)

    if dist is None:
        for j in range(1, n):
            incr = np.linalg.norm(x[j] - x[:j], axis=-1) ** p
            cand = best[:j] + incr
            k = int(np.argmax(cand))
            best[j] = cand[k]
            link[j] = k
    else:
        d = _index_metric(x, dist)
        for j in range(1, n):
            cand = np.array([best[m] + d(m, j) ** p for m in range(j)])
            k = int(np.argmax(cand))
            best[j] = cand[k]
            link[j] = k

    value = float(best[-1] ** (1.0 / p))
    if not return_points:
        return value
    return value, _walk_links(link, n)


def _walk_links(link, n: int) -> list:
    """Follow the predecessor chain from n-1 back to 0."""
    pts = []
    a = n - 1
    while True:
        pts.append(int(a))
        if a == 0:
            break
        a = int(link[a])
    pts.reverse()
    return pts


# ---------------------------------------------------------------------------
# pruned: the Korepanov-Lyons-Zorin-Kranich algorithm
# ---------------------------------------------------------------------------

def p_variation_pruned(x, p: float = 2.0, dist=None, return_points: bool = False):
    r"""Same supremum as `p_variation_exact`, with most candidates skipped.

    The recursion is unchanged. What changes is that the inner scan over
    :math:`m` does not visit every index, by two observations.

    **A threshold, increasing as the scan goes back.** With ``max_pv`` the best
    value found so far for endpoint :math:`j`, candidate :math:`m` improves on
    it only if

    .. math::
        d(x_m, x_j) \;>\; \big(\texttt{max\_pv} - D[m]\big)^{1/p} \;=:\; \delta .

    :math:`D` is non-decreasing, so :math:`\delta` grows as :math:`m` falls:
    older points must be further away to be worth anything.

    **A bound covering a whole block.** For a dyadic block :math:`B` with centre
    :math:`c` and radius :math:`R_B = \max_{m'\in B} d(x_{m'}, x_c)`, the
    triangle inequality gives :math:`d(x_{m'}, x_j) \le R_B + d(x_c, x_j)` for
    every :math:`m' \in B`. If that is at most :math:`\delta`, no member of
    :math:`B` can improve anything and all of them are skipped at once.

    Blocks are dyadic, so each index lies in :math:`\log N` of them and the
    radii are accumulated online as :math:`j` advances. Only the triangle
    inequality is used, so any metric is admissible.

    Cost is about :math:`O(N\log N)` on paths that change direction, and
    :math:`O(N^2)` in the worst case. The worst case is a monotone path in
    :math:`\mathbb{R}`, where :math:`d(m',c) + d(c,j) = d(m',j)` exactly: the
    bound has no slack, so nothing is ever skipped.

    Parameters
    ----------
    x : (T,) or (T, d) array of points.
    p : exponent, p >= 1.
    dist : optional metric on points. Euclidean if omitted. Must satisfy the
        triangle inequality and be symmetric, or the pruning is unsound.
    return_points : also return the maximising subsequence.

    Returns
    -------
    float, or (float, list[int]) if `return_points`.
    """
    p = _check_p(p)
    x = _as_points(x)
    n = x.shape[0]
    if n == 1:
        return (0.0, [0]) if return_points else 0.0

    d = _index_metric(x, dist)

    s = n - 1                              # last index
    levels = max(1, s.bit_length())        # smallest L with s >> L == 0

    # Spatial index. For level k, the block containing j is
    # [ (j >> k) << k , that + 2^k ), its centre is the midpoint, and
    # radius[ind(j, k)] is the largest distance from centre to a member seen so
    # far. Storing at (s >> k) + (j >> k) packs every level into one flat array
    # of length s without collisions.
    radius = [0.0] * s

    def ind(j: int, k: int) -> int:
        return (s >> k) + (j >> k)

    def centre(j: int, k: int) -> int:
        return ((j >> k) << k) + (1 << (k - 1))

    def centre_outside(j: int, k: int) -> bool:
        # True exactly when centre(j, k) > s, i.e. the block is the last one and
        # is truncated below its own midpoint. Guards both the radius index and
        # the distance evaluation.
        return (j >> k) == (s >> k) and ((s >> (k - 1)) & 1) == 0

    run = [0.0] * n                        # run[j] = D[j], unrooted
    link = [0] * n
    max_pv = 0.0

    for j in range(n):
        # Extend the index to cover j, at every level: log N updates.
        for k in range(1, levels + 1):
            if not centre_outside(j, k):
                i = ind(j, k)
                r = d(centre(j, k), j)
                if r > radius[i]:
                    radius[i] = r
        if j == 0:
            continue

        # Seed with the single step from j-1, which is always admissible.
        m = j - 1
        link[j] = m
        delta = d(m, j)
        max_pv = run[m] + delta ** p

        k = 0                              # current block level, carried over
        while m > 0:
            # Largest block starting at m: climb while the low bits are clear.
            while k < levels and (m >> k) & 1 == 0:
                k += 1
            m -= 1

            # Descend levels until some block bound bites, or we reach a point.
            # delta is refreshed lazily: a stale delta is smaller, hence
            # conservative, so the cheap comparison runs first and the root is
            # only taken when it fails.
            delta_is_stale = True
            while k > 0:
                if not centre_outside(m, k):
                    bound = radius[ind(m, k)] + d(centre(m, k), j)
                    if delta >= bound:
                        break
                    if delta_is_stale:
                        gap = max_pv - run[m]
                        delta = gap ** (1.0 / p) if gap > 0.0 else 0.0
                        delta_is_stale = False
                        if delta >= bound:
                            break
                k -= 1

            if k > 0:
                m = (m >> k) << k          # skip the block; loop decrements past it
            else:
                dj = d(m, j)
                if dj >= delta:
                    cand = run[m] + dj ** p
                    if cand >= max_pv:
                        max_pv = cand
                        link[j] = m

        run[j] = max_pv

    value = float(run[-1] ** (1.0 / p))
    if not return_points:
        return value
    return value, _walk_links(link, n)
