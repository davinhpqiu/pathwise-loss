# Literature around response distances on path space

Reviewed 14 October 2026. [Technical appendix](technical.md) owns the proofs;
[README](README.md) owns current status and evidence. This file records
primary-source comparisons relevant to the proofs.
Searches establish specific predecessors and scope differences; they cannot
certify absence of prior work.

## Current contribution boundary

Response-based separation, polynomial identities, Lie developments, log-ODE
error analysis and commutator estimates are established machinery. Candidate
contributions here are quantitative comparisons between worst-case responses
and full weighted Hilbert--Schmidt (HS) certificates, together with explicit
finite-scale feasibility separations. Exact novelty and external correctness
review remain outstanding.

Admissible responses here use real matrix tuples with joint Frobenius budget
$\sum_i\lVert A_i\rVert_F^2\le r^2$ and Euclidean operator output norm.
Each comparison below retains the source's actual norms, decoder, dimension
and quantifiers.

## Direct predecessors

### Developments, separation and polynomial identities

[Chevyrev and Lyons, arXiv:1307.3580v6](https://arxiv.org/html/1307.3580v6),
§2, Lemmas 4.3--4.4, §4.2 and Theorem 4.8, already connect polynomial
identities to separation by matrix developments. Corollary 5.8 relates all
finite compact-group developments to tree-like equivalence for BV paths.
The general operational viewpoint and identity/separation connection have
direct precedent. Fixed response dimension and a quantitative joint norm
budget specify the restricted family used here.

[Nikolaev, Pliska 8 (1986), 65--76](https://www.math.bas.bg/pliska/Pliska-08/Pliska-08-1986-065-076.pdf),
Theorem 1, equation (1), gives the two-variable identity
$[x,y,x,x,[x,y]]=0$ for $\mathfrak{sl}_2$ in characteristic zero,
using left-nested brackets. This equals our $J_6$. The paper also records
absence of two-variable identities below degree six. Our pure-path blindness
example applies classical algebra; neither identity nor threshold is new.

[Kanel-Belov, Malev and Rowen, arXiv:1506.06792v2](https://arxiv.org/html/1506.06792v2),
Lemma 1, Corollary 1 and Question 3, treat Lie identities of matrix algebras,
including the unrestricted degree-five threshold at dimension two.
Associative Amitsur--Levitzki identities do not automatically give Lie
identities. Number of variables must accompany any threshold statement.

General matrices do not evaluate a finite-step nilpotent quotient: discarded
BCH terms can survive evaluation. Full blindness examples here use full
signature series of pure weakly geometric rough paths, with $p\ge6$.
Existence of a nontrivial BV path with the same fixed-dimension blindness is
not proved by these examples.

### Sharp commutators and product formulas

[Böttcher and Wenzel, Linear Algebra Appl. 429 (2008), 1864--1885](https://doi.org/10.1016/j.laa.2008.05.020)
prove $\lVert[A,B]\rVert_F\le\sqrt2\lVert A\rVert_F\lVert B\rVert_F$.
Repeated application plus scalar optimization proves our Engel constant;
a diagonal/nilpotent block attains it. Present this as a corollary.

[Fox, arXiv:2610.11800v1](https://arxiv.org/html/2610.11800v1),
posted 8 October 2026, Theorems 1.1--1.2 and Example 1, gives sharp
representation-dependent commutator bounds and recovers Böttcher--Wenzel
in the defining matrix representation. Restricted Lie algebra constants
cannot be substituted into our unrestricted real matrix-tuple class.

[Iserles and Kropielnicka, arXiv:2401.06635v1](https://arxiv.org/html/2401.06635v1),
Theorem 2.1 and §3.1, gives the commutator integral used in our two-factor
merge proof and Lie--Trotter error estimates. Our exponential response
upper bound is a standard consequence. The quantitative comparison with
the Gram-derived HS certificate is the candidate addition.

### Lie developments and sharp signature tails

[Boedihardjo, Geng and Souris, arXiv:1811.12170v2](https://arxiv.org/html/1811.12170v2),
Definition 4.1, §4.2.2, §4.2.4 and §5, studies spectral Lie-polynomial
evaluations, with explicit low-degree calculations. Their driver $\ell^1$
norm and Banach operator constraints differ from our joint Frobenius budget.
This is direct adjacent gauge work; exact constants require translating norms.

[Li and Gao, arXiv:2606.30055v1](https://arxiv.org/html/2606.30055v1),
Proposition 4.2 and Theorems 4.3, 4.6, builds contractive developments and
proves pure rough-path tail asymptotics for reasonable tensor norms.
Fixed-degree constrained operator maxima and tail limits as degree tends
to infinity are distinct comparisons. Eigenvalue-based lower witnesses
are established technique.

[Boedihardjo, Geng and Wang, arXiv:2506.18207v1](https://arxiv.org/html/2506.18207v1),
Theorems 3.1, 4.1 and 5.1, addresses the Lyons--Sidorova conjecture through
Cartan development and log-signature convergence. These statements concern
signature tails and regularity, rather than our small-scale certificate ratio.

### FPL criterion and storage

[Ferrucci, Perrée and Lyons, arXiv:2607.26281v1](https://arxiv.org/html/2607.26281v1),
(1.9)--(1.10), Theorem 1.3 and (1.13), supplies the norm/response criterion,
length envelope and scalar example. The paragraph after (2.10) already
identifies the first discarded log-signature as leading local log-ODE error.
Proposition 2.10 requires growth of $m$ in the oscillation parameter
$\varrho$, not in degree $\nu$. Figures 8--9 use empirical averaged errors,
not worst-case certificate feasible regions. The arXiv record still lists
v1 on 14 October 2026.

Our exact BV minimax proof promotes their scalar example to a universal
upper bound, including path-dependent partitions. It improves the additional
length envelope, while a straight line attains its exact HS certificate.
Leading BV storage asymptotics persist. Pure Engel storage brackets show
response/HS conservatism only within fixed Euler decoding and generic dense
storage; a single pure log block has exact log-ODE response.

### Log-ODE error and adaptivity

[Boutaib, Gyurkó, Lyons and Yang, arXiv:1307.4708v1](https://arxiv.org/html/1307.4708v1),
Theorem 18 and Remark 16, provides dimension-free Lie-series/Euler estimates.
These are a starting point for a rough-input extension, with vector-field
regularity and rough controls replacing BV length.

[Bayer, Breneis and Lyons, arXiv:2307.12590v1](https://arxiv.org/html/2307.12590v1),
Theorem 3.1 and Lemmas 4.2--4.3, provides fixed-system global/local error
analysis and adaptive log-ODE selection. Our fixed-partition BV result adds
a norm-constrained response/HS comparison. Existing adaptivity machinery
does not establish novelty of that comparison.

## Nearby problems with different quantifiers

### Signature model reduction and bilinear realization

[Bayer and Redmann, arXiv:2412.14723v2](https://arxiv.org/html/2412.14723v2),
Assumption 3.1, equations (3.1)--(3.4), Propositions 3.4--3.5, reduces
a fixed time-extended Brownian signature system for a specified output,
using expected system Gramians. Path compression uniform over all admissible
coefficient tuples has different objectives and quantifiers. Equivalence
of the two optimization problems is unproved.

[Petreczky, ESAIM COCV 17 (2011), Part II](https://www.numdam.org/item/10.1051/cocv/2010015.pdf),
equations (2.3), (2.6), Theorem 2.3, gives Fliess-series coefficients and
minimal bilinear switched realizations under arbitrary switching, with
Hankel-rank characterization and uniqueness up to isomorphism. Restricted
switching languages require qualified conclusions (Theorem 2.4).
This is fixed-system realization, rather than fixed-dimensional blindness
for a family of systems.

### Path characteristic distances and neural developments

[Lou, Li and Ni, PCF-GAN, arXiv:2305.12511v2](https://arxiv.org/html/2305.12511v2),
Definition 3.3, averages squared HS differences of expected unitary
developments. Dirac laws already give deterministic path comparisons.
Thus “laws versus individual paths” cannot justify novelty. Our unrestricted
real matrices, worst-case rather than sampled response, operator output norm
and joint budget are the relevant differences.

[Walker et al., arXiv:2402.18512v4](https://arxiv.org/html/2402.18512v4),
§§3.1--3.3 and Theorem 3.1, uses nonlinear vector fields and iterated brackets
in log neural CDEs, with regularity controlled by weights. A linear Frobenius
tuple budget alone does not certify those nonlinear models' errors.
The connection supplies motivation, not an immediate application theorem.

### Homogeneous geometry and matrix optimization

[Le Donne, arXiv:1604.08579v1](https://arxiv.org/html/1604.08579v1),
Definition 2.1 and §2.3, defines homogeneous distances using graded dilations.
A pure degree-$k$ amplitude $\varepsilon L$ has Carnot scale
$|\varepsilon|^{1/k}$; our fixed-budget response has scale $|\varepsilon|$.
Our tangent lemma is uniform first-order matrix-exponential analysis.
A Carnot homogeneous-metric interpretation requires additional work.

[Briët, Regev and Saket, arXiv:1412.4413v2](https://arxiv.org/html/1412.4413v2),
introductory optimization and hardness theorem, uses bilinear unitary
matrix variables. A reduction to our repeated-variable polynomial on a
joint Frobenius ball has not been supplied. Computational hardness here
remains unproved. Smith's amplification lemma also requires a norm-preserving
linearization before it could address dimension-two sufficiency.

## Review access and limits

Primary theorem sections listed above were read; complete proofs of every
cited paper were not independently audited. Nikolaev's scanned original
was checked visually at journal page 65. The Böttcher--Wenzel original
publication metadata and statement were checked, with its matrix constant
also confirmed in Fox; full original proof was not re-audited.

Lyons--Sidorova's sound-compression paper remains background: the author PDF
returned an access error during this review. Smith and several older surveys
were not freshly read. Their contents are not used to establish a new claim.
Broad learning-rate, kernel and Sobolev-training literature is background
rather than evidence about the exact constants or merge theorem.

External review of our arguments, particularly infinite-level all-scale
conditions and BV perturbation bounds, remains outstanding. Literature
searches did not identify an exact match for the current constrained ratios
and finite-scale comparisons, but this does not certify novelty.
