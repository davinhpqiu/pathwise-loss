# Technical appendix

Consolidated 14 October 2026. Complete working arguments, with one source
for definitions, gauges, merge results, storage and planar BV proofs.
The shorter sending exposition is [note.tex](note.tex); prior work and novelty
limits are in [literature.md](literature.md). External expert validation
remains outstanding. Historical derivations, designs and audits are preserved
in the dated ZIP under `archive/`, not maintained as parallel documents.

- [Definitions](#definitions)
- [Directional gauges, identities and homogeneous merges](#gauges)
- [Euler minimax and storage](#storage)
- [Planar BV and mixed displacement/area blocks](#planar-bv)

<a id="definitions"></a>
## A. Definitions and conventions

### 2. Paths, signatures and norms

Unless another class is stated, $X:[0,T]\to\mathbb R^d$ is continuous and
of bounded variation (BV). For a word $w=(w_1,\ldots,w_k)$,

$$
S^w(X)=\int_{0<u_1<\cdots<u_k<T}
dX^{w_1}_{u_1}\cdots dX^{w_k}_{u_k},\qquad
S^{(k)}(X)=\sum_{|w|=k}S^w(X)e_w.
$$

Word tensors $e_w=e_{w_1}\otimes\cdots\otimes e_{w_k}$ are orthonormal:
$\lVert D_k\rVert_{\mathrm{HS}}^2=\sum_{|w|=k}|D_k^w|^2$.
Set $D_k=S^{(k)}(X)-S^{(k)}(\widetilde X)$ for a path pair.
For other decoded series, $D_k$ denotes their specified degree-$k$ difference.

Starting points require separate anchoring when informative. Signatures are
translation invariant and ignore increasing reparameterization; time
augmentation retains timing. Rough-path statements specify their class and
use full extended signatures.

### 3. Linear response

For response dimension $n$, use real matrices $A=(A_1,\ldots,A_d)$ with joint
Frobenius norm $|A|^2=\sum_i\lVert A_i\rVert_F^2$. Row states follow

$$
dY_t=\sum_iY_tA_i\,dX_t^i,\qquad
Y_T=Y_0\Phi^A(X),\qquad
\Phi^A(X)=I_n+\sum_{k\ge1}R_{A,k}(S^{(k)}(X)),
$$

where $R_{A,k}(D)=\sum_{|w|=k}D^wA_{w_1}\cdots A_{w_k}$.
Outputs use Euclidean operator norm. Matrix products are chronological;
transposing tuples converts to column-state conventions with unchanged suprema.
BV factorial bounds ensure absolute convergence.

### 4. Exact response discrepancy

For coefficient radius $r\ge0$,

$$
d_{r,n}(X,\widetilde X)=\sup_{|A|\le r}
\lVert\Phi^A(X)-\Phi^A(\widetilde X)\rVert_{\mathrm{op}}.
$$

This is a pseudometric on classes with finite uniform responses.
It increases with $r$ and $n$; zero-block embedding proves dimension
monotonicity. For real dilation $c$, $S^{(k)}(\delta_cX)=c^kS^{(k)}(X)$ and

$$
d_{r,n}(\delta_cX,\delta_c\widetilde X)
=d_{|c|r,n}(X,\widetilde X).
$$

For BV paths the null set is independent of $r>0$, by analytic continuation
in $A$. Within the convergent small-coefficient logarithm expansion,
equal responses force each homogeneous log-signature difference into the
common kernel of all matrix evaluations. This is the classical
identity/development connection (Chevyrev--Lyons, §4.2).

A homogeneous Lie identity generates a fully blind pure weakly geometric
$p$-rough path when $p$ is at least its degree. At dimension two,
$m_2(2)=6$ (Nikolaev) and $m_2(d)=5$ for $d\ge3$.
The supplied pure example proves no nontrivial BV blindness example.
General matrices do not evaluate a finite-step nilpotent quotient independently
of discarded BCH terms. See [proofs §6](technical.md#61-a-strict-response-dimension-separation-at-degree-six).

### 5. Truncated response discrepancy

For depth $L$,

$$
d^{(\le L)}_{r,n}(D)=\sup_{|A|\le r}
\left\lVert\sum_{k=1}^LR_{A,k}(D_k)\right\rVert_{\mathrm{op}}.
$$

One tuple acts across every retained degree. This can differ from the sum
of independently optimized level responses.

### 6. Level-wise response discrepancy

For homogeneous $D_k$, define

$$
\gamma_{k,n}(D_k)=\sup_{|A|\le1}\lVert R_{A,k}(D_k)\rVert_{\mathrm{op}},
\qquad d^{(k)}_{r,n}(D_k)=r^k\gamma_{k,n}(D_k).
$$

For $D_k\ne0$, set $\rho_{k,n}(D_k)=\gamma_{k,n}(D_k)/
\lVert D_k\rVert_{\mathrm{HS}}\in[0,1]$, and set $\rho_{k,n}(0)=0$.
The ratio is invariant under nonzero rescaling and simultaneous orthogonal
driver changes. It depends on tensor direction and response dimension.

### 7. Response bounds and certificates

Coefficientwise Cauchy--Schwarz gives

$$
d^{(\le L)}_{r,n}(D)
\le\sum_{k=1}^Lr^k\gamma_{k,n}(D_k)
\le\sum_{k=1}^Lr^k\lVert D_k\rVert_{\mathrm{HS}}
\le\sum_{k=1}^Lr^k\lVert D_k\rVert_1.
$$

Within-level distortion and incompatible maximizing tuples across levels are
separate sources of slack. Numerical lower estimates alone cannot prove a
strict gap between exact suprema.

For full series $F,G$ with convergent certificate, define

$$
\Delta_{r,n}(F,G)=\sup_{|A|\le r}
\lVert R_A(F)-R_A(G)\rVert_{\mathrm{op}},\qquad
\mathcal B_r(F,G)=\sum_{k\ge1}r^k\lVert F_k-G_k\rVert_{\mathrm{HS}}.
$$

A certificate is sufficient for response accuracy. Further length/control
bounds on $\mathcal B_r$ are another layer of conservatism.

### 8. Constructive lower bounds

For $n\ge k+1$ and unit vectors $u_1,\ldots,u_k\in\mathbb R^d$, place their
coordinates on shifts:

$$
A_i=k^{-1/2}\sum_{j=1}^k(u_{k+1-j})_iE_{j+1,j}.
$$

Then $|A|=1$ and the $(k+1,1)$ entry of $R_{A,k}(D)$ is
$k^{-k/2}\langle D,u_1\otimes\cdots\otimes u_k\rangle$. Thus

$$
\gamma_{k,n}(D)\ge k^{-k/2}\lVert D\rVert_\varepsilon
\ge k^{-k/2}\max_w|D^w|,
$$

where $\lVert D\rVert_\varepsilon$ is the supremum of the displayed rank-one
contraction over unit vectors. These bounds guarantee visibility at sufficient
dimension but need not be sharp.

### 9. Exact benchmarks and tangent lemma

For $\lVert u\rVert_2=1$, let $E_k=\operatorname{ad}_{e_1}^{k-1}(e_2)$ and
$G_4=[e_1,[e_2,[e_1,e_2]]]$.

| direction | exact $\rho$ | response dimension |
|---|---|---|
| level one; $c u^{\otimes k}$ | $1$ | $n\ge1$ |
| $d^{-1/2}\sum_i e_i\otimes e_i$ | $d^{-1/2}$ | $n\ge1$ |
| nonzero planar area | $1/2$ | $n\ge2$ |
| nonzero planar cubic Lie tensor | $2\sqrt2/9$ | $n\ge2$ |
| $E_k$, $k\ge2$ | $[2(k-1)]^{(k-1)/2}/(k^{k/2}\sqrt{\binom{2k-2}{k-1}})$ | $n\ge2$ |
| $G_4$ | $\sqrt{(411+41\sqrt{41})/20480}$ | $n=2$ |

Level-one and symmetric cases follow by scalar/rank-one attainment. Detailed
Lie proofs are in [proofs §§2--4](technical.md#2-exact-formula-for-an-engel-direction).
For a single word with letter counts $m_i$,
$\rho_{k,n}(e_w)=\prod_{i:m_i>0}(m_i/k)^{m_i/2}$ for every $n\ge1$
([§6](technical.md#6-single-words)).

For homogeneous Lie $L$ of degree $k$, the pure path
$\mathbf X_t^\varepsilon=\exp_\otimes(t\varepsilon L)$ requires $p\ge k$.
Writing $C=r^k\lVert L\rVert_{\mathrm{HS}}$,

$$
\left|\frac{d_{r,n}(\mathbf X^\varepsilon,\mathbf1)}{|\varepsilon|}
-r^k\gamma_{k,n}(L)\right|
\le\frac{|\varepsilon|C^2}{2}e^{|\varepsilon|C}.
$$

For $k\le N<2k$ the truncated response is exactly
$|\varepsilon|r^k\gamma_{k,n}(L)$.
This is uniform first-order exponential analysis; Carnot amplitude scaling is
$|\varepsilon|^{1/k}$. See [proofs §7](technical.md#7-the-response-gauge-is-the-tangent-path-discrepancy).

### 10. Partitioned decoders

On partition $\pi$, degree-$N$ Euler and log-ODE decoders are respectively

$$
\mathcal E^A_{N,\pi}(X)=\prod_i
\left[I+\sum_{k=1}^NR_{A,k}(S(X_i)_k)\right],
\qquad
\mathcal M^A_{N,\pi}(X)=\prod_i
\exp\left(\sum_{k=1}^NR_{A,k}(\ell_{i,k})\right),
$$

where $\ell_{i,k}$ is local homogeneous log-signature. The same tuple acts on
all factors. Euler products can generate degrees higher than $N$; discarded
degrees and stored log depth are distinct.

Define composed discrepancy $\widetilde d$ by replacing exact responses in
$d_{r,n}$ with the chosen decoder. If each path's uniform decoder error is
at most $\varepsilon_X,\varepsilon_{\widetilde X}$, then

$$
|d_{r,n}(X,\widetilde X)-\widetilde d(X,\widetilde X)|
\le\varepsilon_X+\varepsilon_{\widetilde X}.
$$

Comparing two approximate representations requires their original-path errors.
Exact decoded-factor merge theorems do not remove that requirement.


<a id="gauges"></a>
## B. Directional gauges and homogeneous merge proofs

### 1. Response gauge and orthogonal invariance

For a homogeneous tensor $D\in(\mathbb R^d)^{\otimes k}$, write

$$
\gamma_{k,n}(D)
=
\sup_{\sum_i\lVert A_i\rVert_F^2\le1}
\left\lVert
\sum_{i_1,\ldots,i_k}
D^{i_1\cdots i_k}A_{i_1}\cdots A_{i_k}
\right\rVert_{\mathrm{op}}.
$$

Thus $\rho_{k,n}(D)=\gamma_{k,n}(D)/\lVert D\rVert_{\mathrm{HS}}$
when $D\ne0$.

The gauge is invariant under simultaneous orthogonal changes of driver
coordinates.  If $Q\in O(d)$ and $QD=Q^{\otimes k}D$, then

$$
\gamma_{k,n}(QD)=\gamma_{k,n}(D).
$$

Indeed, evaluation of $Q^{\otimes k}D$ at $(A_1,\ldots,A_d)$ equals evaluation
of $D$ at the orthogonally mixed tuple

$$
B_j=\sum_i Q_{ij}A_i.
$$

The map $A\mapsto B$ is a bijection of the Frobenius unit ball because
$\sum_j\lVert B_j\rVert_F^2=\sum_i\lVert A_i\rVert_F^2$.  Hilbert--Schmidt norm
is invariant under the same action, so $\rho_{k,n}$ is invariant as well.

### 2. Exact formula for an Engel direction

#### Theorem

Let $d\ge2$, $n\ge2$, $k\ge2$, and

$$
E_k=\operatorname{ad}_{e_1}^{k-1}(e_2).
$$

Then

$$
\boxed{
\gamma_{k,n}(E_k)
=
\frac{[2(k-1)]^{(k-1)/2}}{k^{k/2}}
}
$$

and

$$
\boxed{
\rho_{k,n}(E_k)
=
\frac{[2(k-1)]^{(k-1)/2}}
{k^{k/2}\sqrt{\binom{2k-2}{k-1}}}.
}
$$

For $n=1$, $\gamma_{k,1}(E_k)=0$ because scalar matrices commute.

#### Proof: upper bound

Write $A=A_1$, $B=A_2$, $a=\lVert A\rVert_F$, and
$b=\lVert B\rVert_F$.  Evaluation of $E_k$ is the iterated commutator

$$
\operatorname{ad}_A^{k-1}(B).
$$

The Böttcher--Wenzel inequality (Böttcher and Wenzel, 2008) gives

$$
\lVert[X,Y]\rVert_F
\le\sqrt2\lVert X\rVert_F\lVert Y\rVert_F.
$$

Iteration and the scalar optimization below give this corollary of
Böttcher--Wenzel.

Applying it $k-1$ times, and using operator norm at most Frobenius norm,

$$
\begin{aligned}
\left\lVert\operatorname{ad}_A^{k-1}(B)\right\rVert_{\mathrm{op}}
&\le
\left\lVert\operatorname{ad}_A^{k-1}(B)\right\rVert_F\\
&\le
(\sqrt2)^{k-1}a^{k-1}b.
\end{aligned}
$$

The constraint is $a^2+b^2\le1$; unused matrices can only consume norm, so
they may be set to zero.  Homogeneity puts the maximizer on $a^2+b^2=1$.
Maximizing $a^{k-1}b$ gives

$$
a^2=\frac{k-1}{k},
\qquad
b^2=\frac1k.
$$

Consequently

$$
\gamma_{k,n}(E_k)
\le
(\sqrt2)^{k-1}
\left(\frac{k-1}{k}\right)^{(k-1)/2}
\frac1{\sqrt k}
=
\frac{[2(k-1)]^{(k-1)/2}}{k^{k/2}}.
$$

#### Proof: attainment

It is enough to use a $2\times2$ block.  Set

$$
A=
\frac{a}{\sqrt2}
\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad
B=b
\begin{pmatrix}0&1\\0&0\end{pmatrix},
$$

with $a^2=(k-1)/k$ and $b^2=1/k$.  These matrices have Frobenius norms $a$
and $b$, and

$$
\operatorname{ad}_A^{k-1}(B)
=(\sqrt2a)^{k-1}b
\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$

The last matrix has equal Frobenius and operator norms.  Hence equality holds
at every step of the upper bound.  Embedding this block in the top-left corner
proves attainment for every $n\ge2$.

Here $B$ is an eigenvector of $\operatorname{ad}_A$ with eigenvalue
$\sqrt2\lVert A\rVert_F$ inside an $\mathfrak{sl}_2$ block. Li and Gao (2026)
use the same device at the level of tail asymptotics, building for every
top-level tensor a contractive development in which its tensor norm appears as
an eigenvalue; their result proves Conjecture 2.12 of Boedihardjo, Geng and
Souris (2020) for arbitrary reasonable tensor norms. Present statement differs
in fixing degree and constraining coefficient norm, rather than taking degree to
infinity.

#### Proof: tensor norm

Expanding the iterated commutator gives

$$
E_k
=
\sum_{j=0}^{k-1}
(-1)^j\binom{k-1}{j}
e_1^{\otimes(k-1-j)}\otimes e_2\otimes e_1^{\otimes j}.
$$

The displayed word tensors are distinct and orthonormal.  Vandermonde's
identity therefore gives

$$
\lVert E_k\rVert_{\mathrm{HS}}^2
=
\sum_{j=0}^{k-1}\binom{k-1}{j}^2
=
\binom{2k-2}{k-1}.
$$

Dividing the response value by this norm proves the ratio formula.

### 3. Planar consequences at levels two and three

At level two, the planar free Lie component is one-dimensional and spanned by
$E_2=[e_1,e_2]$.  The theorem yields

$$
\gamma_{2,n}(E_2)=\frac1{\sqrt2},
\qquad
\lVert E_2\rVert_{\mathrm{HS}}=\sqrt2,
\qquad
\rho_{2,n}=\frac12.
$$

At level three, a basis is

$$
E_3=[e_1,[e_1,e_2]],
\qquad
F_3=[e_2,[e_1,e_2]].
$$

The tensors are orthogonal and each has squared Hilbert--Schmidt norm six.
Every nonzero planar degree-three Lie element has the form

$$
uE_3+vF_3=[ue_1+ve_2,[e_1,e_2]].
$$

A rotation in $SO(2)$ sends $(u,v)$ to $(\sqrt{u^2+v^2},0)$ and fixes the
oriented area element $[e_1,e_2]$.  Orthogonal invariance and homogeneity thus
reduce every nonzero element to $E_3$.  The theorem gives the exact constant

$$
\boxed{
\rho_{3,n}(L)=\frac{2\sqrt2}{9}
\quad
\text{for every }0\ne L\in\mathcal L_3(\mathbb R^2),\ n\ge2.
}
$$

Scalar responses give zero on both levels because they annihilate all Lie
brackets.

### 4. Level four: Engel and mixed directions

Planar degree-four Lie component $\mathcal L_4(\mathbb R^2)$ has dimension 3,
spanned by $E_4$, its image under $e_1\leftrightarrow e_2$, and one element of
multidegree $(2,2)$.

#### 4.1 Engel direction

For

$$
E_4=[e_1,[e_1,[e_1,e_2]]],
$$

theorem of §2 gives

$$
\gamma_{4,n}(E_4)=\frac{3\sqrt6}{8},
\qquad
\lVert E_4\rVert_{\mathrm{HS}}=\sqrt{20},
\qquad
\boxed{
\rho_{4,n}(E_4)=\frac{3\sqrt{30}}{80}
}
\quad(n\ge2).
$$

Swapping $e_1,e_2$ is orthogonal, so the $(1,3)$ Engel direction has same ratio.

#### 4.2 Mixed direction

Let

$$
G_4=[e_1,[e_2,[e_1,e_2]]]
=2e_{1212}-e_{1122}-2e_{2121}+e_{2211},
\qquad
\lVert G_4\rVert_{\mathrm{HS}}=\sqrt{10},
$$

with $e_{ijkl}=e_i\otimes e_j\otimes e_k\otimes e_l$.

##### Theorem

For $d\ge2$,

$$
\boxed{
\gamma_{4,2}(G_4)=\sqrt{\frac{411+41\sqrt{41}}{2048}}\approx0.573473,
\qquad
\rho_{4,2}(G_4)=\sqrt{\frac{411+41\sqrt{41}}{20480}}\approx0.181348.
}
$$

Hence $\rho_{4,2}(G_4)<\rho_{4,2}(E_4)$: for $n=2$, $\rho_{4,2}$ is not constant on
$\mathcal L_4(\mathbb R^2)\setminus\{0\}$. For $n\ge3$, embedding in a
$2\times2$ block gives only $\gamma_{4,n}(G_4)\ge\gamma_{4,2}(G_4)$. Ascent at
$n=3,4$ (150 restarts each) returns same value to $10^{-9}$; matching upper bound
is unproved, so level-four anisotropy for $n\ge3$ remains numerical.

##### Proof

*Step 1: reduction to $\mathfrak{sl}_2$.* Only $A=A_1$, $B=A_2$ enter, so other
$A_i$ are set to zero. Scalar parts leave commutators unchanged and add to
Frobenius norm, so $A,B$ are traceless. On traceless $2\times2$ matrices
$XY+YX=\operatorname{tr}(XY)I$, which gives

$$
[A,[B,[A,B]]]=2\operatorname{tr}(AB)[A,B].
$$

Objective is homogeneous of degree 4, so

$$
\gamma_{4,2}(G_4)
=\sup_{\substack{\operatorname{tr}A=\operatorname{tr}B=0\\ \lVert A\rVert_F^2+\lVert B\rVert_F^2=1}}
2\lvert\operatorname{tr}(AB)\rvert\,\lVert[A,B]\rVert_{\mathrm{op}}.
$$

*Step 2: coordinates.* Frobenius-orthonormal basis of $\mathfrak{sl}_2(\mathbb R)$:

$$
H=\tfrac1{\sqrt2}\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
S=\tfrac1{\sqrt2}\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
K=\tfrac1{\sqrt2}\begin{pmatrix}0&1\\-1&0\end{pmatrix}.
$$

Write $A=a_1H+a_2S+a_3K$, $B=b_1H+b_2S+b_3K$, so
$\lVert A\rVert_F=|a|$, $\lVert B\rVert_F=|b|$. Let $z=a\times b$ (cross product in
$\mathbb R^3$) and $M=\operatorname{diag}(1,1,-1)$. From
$\operatorname{tr}H^2=\operatorname{tr}S^2=1$, $\operatorname{tr}K^2=-1$ and
$[H,S]=\sqrt2K$, $[H,K]=\sqrt2S$, $[S,K]=-\sqrt2H$,

$$
\operatorname{tr}(AB)=a^{\top}Mb,
\qquad
[A,B]=\sqrt2\,(-z_1H-z_2S+z_3K).
$$

For traceless $N=xH+yS+wK$, $\lVert N\rVert_F^2=x^2+y^2+w^2$ and
$\det N=-(x^2+y^2-w^2)/2$, so its singular values satisfy
$\sigma_{\max}^2=(\sqrt{x^2+y^2}+|w|)^2/2$. Therefore

$$
\lVert[A,B]\rVert_{\mathrm{op}}=\sqrt{z_1^2+z_2^2}+|z_3|.
$$

*Step 3: fix plane.* Objective vanishes when $z=0$. Otherwise let
$V=\operatorname{span}\{a,b\}$, $\nu=z/|z|$ its unit normal, $\tau=\nu_3^2\in[0,1]$.
Then $\lVert[A,B]\rVert_{\mathrm{op}}=|z|(\sqrt{1-\tau}+\sqrt\tau)$. Restricted
to $V$, form $x^{\top}My$ has matrix $I-2gg^{\top}$ in an orthonormal basis,
$g$ being projection of $e_3$ onto $V$, $|g|^2=1-\tau$; its eigenvalues are 1 and
$\mu=2\tau-1$. Let $Z\in\mathbb R^{2\times2}$ hold coordinates of $a,b$ as columns
in a corresponding eigenbasis and $Q=\operatorname{diag}(1,\mu)$. Then

$$
|z|=|\det Z|,
\qquad
a^{\top}Mb=(Z^{\top}QZ)_{12},
\qquad
\lVert Z\rVert_F=1.
$$

Every plane, hence every $\tau\in[0,1]$, occurs.

*Step 4: optimise $Z$.* Write $Z=R_\psi\Sigma W$ with $R_\psi$ a rotation,
$\Sigma=\operatorname{diag}(\sigma_1,\sigma_2)$, $W\in O(2)$ (a left reflection
$\operatorname{diag}(1,-1)$ commutes with $\Sigma$ and moves right). For symmetric
$T=\begin{pmatrix}x&y\\y&w\end{pmatrix}$,
$\max_{W\in O(2)}|(W^{\top}TW)_{12}|=\sqrt{y^2+(x-w)^2/4}$. Apply this to
$T=\Sigma R_\psi^{\top}QR_\psi\Sigma$ with $m=1-\mu=2(1-\tau)$, $\xi=\sin^2\psi$,
$s=\sigma_1^2$, $P=s(1-s)$:

$$
x=s(1-m\xi),\quad
w=(1-s)(1-m+m\xi),\quad
y^2=P\,m^2\xi(1-\xi),
$$

$$
\det(Z)^2\bigl(y^2+(x-w)^2/4\bigr)
=P\Bigl[Pm^2\xi(1-\xi)+\tfrac14\bigl(2s-1+(1-s)m-m\xi\bigr)^2\Bigr].
$$

Coefficient of $\xi^2$ is $m^2P(\tfrac14-P)\ge0$, so maximum over $\xi\in[0,1]$
is at an endpoint. Endpoint $\xi=1$ with $s$ equals $\xi=0$ with $1-s$. At
$\xi=0$, with $r=1-s$, $2s-1+(1-s)m=1-2\tau r$. Combining with step 3,

$$
\gamma_{4,2}(G_4)^2=\sup_{\tau,r\in[0,1]}\Psi(\tau,r),
\qquad
\Psi=\bigl(1+2\sqrt{\tau(1-\tau)}\bigr)\,r(1-r)\,(1-2\tau r)^2 .
$$

*Step 5: maximise $\Psi$.* Using $1+2\sqrt{\tau(1-\tau)}\le2$ and
$\max_{r}r(1-r)(1-2r)^2=1/16$:

- if $1-2\tau r<0$ then $\tau,r>\tfrac12$ and $0<2\tau r-1\le2r-1$, so $\Psi\le1/8$;
- on $\tau=0$, $\Psi\le1/4$; on $\tau=1$, $\Psi\le1/16$; on $r\in\{0,1\}$ or
  $1-2\tau r=0$, $\Psi=0$.

Point below gives $\Psi\approx0.3289>1/4$, so maximum lies in open set
$U=\{0<\tau<1,\,0<r<1,\,1-2\tau r>0\}$, where $\Psi$ is smooth and positive. With
$q=\sqrt{\tau(1-\tau)}$, stationarity of $\log\Psi$ reads

$$
8\tau r^2-(6\tau+2)r+1=0,
\qquad
(1-2\tau)(1-2\tau r)=4r\bigl(q+2\tau(1-\tau)\bigr).
$$

Second equation fixes $q$ rationally; imposing $q^2=\tau(1-\tau)$ and taking
resultant in $r$ with first equation gives

$$
16\tau(2\tau-1)^3(50\tau^2-53\tau+4)=0.
$$

At $\tau=\tfrac12$ left side of second equation is 0 and right side positive.
At $\tau=(53+7\sqrt{41})/100$ the two roots $r$ give $1-2\tau r<0$ or $q<0$. At
$\tau=(53-7\sqrt{41})/100$ one root exceeds 1; the other is admissible. Unique
critical point in $U$:

$$
\tau_*=\frac{53-7\sqrt{41}}{100},\quad
r_*=\frac{21-\sqrt{41}}{32},\quad
q_*=\frac{21+\sqrt{41}}{100},\quad
1-2\tau_*r_*=\frac{1+\sqrt{41}}{8}.
$$

Then $1+2q_*=(71+\sqrt{41})/50$, $r_*(1-r_*)=(190+10\sqrt{41})/1024$,
$(1-2\tau_*r_*)^2=(42+2\sqrt{41})/64$, and

$$
\Psi(\tau_*,r_*)
=\frac{1600\,(411+41\sqrt{41})}{50\cdot1024\cdot64}
=\frac{411+41\sqrt{41}}{2048}.
$$

Each supremum is over a compact set, so value is attained by $2\times2$
matrices. Dividing by $\lVert G_4\rVert_{\mathrm{HS}}=\sqrt{10}$ gives $\rho$.
$\square$

Resultant and admissibility checks are exact symbolic computations; value
agrees with direct ascent over $2\times2$, $3\times3$ and $4\times4$ tuples.

### 5. Asymptotic consequence

Put $m=k-1$.  The central binomial coefficient satisfies

$$
\binom{2m}{m}\sim\frac{4^m}{\sqrt{\pi m}}.
$$

The exact Engel ratio therefore obeys

$$
\rho_{k,n}(E_k)
\sim
\pi^{1/4}e^{-1/2}
2^{-(k-1)/2}(k-1)^{-1/4}.
$$

Along this explicit family, the Hilbert--Schmidt upper bound becomes
exponentially more conservative with signature level even though a
$2\times2$ response attains the exact response supremum at every level.

| $k$ | $\rho_{k,n}(E_k)$ | overestimate $1/\rho$ |
|---:|---:|---:|
| 2 | 0.500000 | 2.00 |
| 3 | 0.314270 | 3.18 |
| 4 | 0.205396 | 4.87 |
| 5 | 0.136838 | 7.31 |
| 6 | 0.092224 | 10.84 |
| 8 | 0.042787 | 23.37 |
| 10 | 0.020198 | 49.51 |

### 6. Single words

For a word $w$ of length $k$ with $m_i$ occurrences of letter $i$,

$$
\boxed{
\rho_{k,n}(e_{w_1}\otimes\cdots\otimes e_{w_k})=\prod_{i=1}^d\left(\frac{m_i}{k}\right)^{m_i/2}
\quad(n\ge1),
}
$$

with $0^0=1$.

*Proof.* Let $a_i=\lVert A_i\rVert_F$. Then
$\lVert A_{w_1}\cdots A_{w_k}\rVert_{\mathrm{op}}\le\prod_ja_{w_j}=\prod_ia_i^{m_i}$.
Maximising $\sum_im_i\log a_i$ subject to $\sum_ia_i^2=1$ gives $a_i^2=m_i/k$.
Equality holds for $A_i=a_iP$, $P$ a rank-one orthogonal projection, since
$P^k=P$ and $\lVert P\rVert_{\mathrm{op}}=\lVert P\rVert_F=1$. A word tensor
has Hilbert--Schmidt norm 1. $\square$

Examples: $e_1\otimes e_2$ gives $1/2$, $e_{121}$ gives $2/(3\sqrt3)\approx0.3849$,
$e_{1212}$ gives $1/4$, matching single-word rows of `README.md#retained-evidence`.

#### 6.1 A strict response-dimension separation at degree six

Define the two-letter homogeneous Lie polynomial

$$
J_6=[E_4,E_2]
=[\operatorname{ad}_{e_1}^{3}(e_2),[e_1,e_2]].
$$

This is Nikolaev's classical two-variable identity: Theorem 1, equation (1),
*Pliska Studia Math. Bulgarica* 8 (1986), 65--76. His left-normed expression
$[x,y,x,x,[x,y]]$ equals the displayed polynomial. The response-dimension
example below applies that identity; its algebraic form and degree-six
two-variable threshold are established prior work.

Then

$$
\gamma_{6,2}(J_6)=0,
\qquad
\gamma_{6,3}(J_6)\ge\frac6{343}>0.
$$

*Proof.* Scalar parts drop out of commutators. For traceless $2\times2$
matrix $A_0$, Cayley--Hamilton gives
$A_0^2=\tfrac12\operatorname{tr}(A_0^2)I$ and hence

$$
\operatorname{ad}_{A_0}^{3}(B)
=2\operatorname{tr}(A_0^2)[A_0,B].
$$

The two factors in $J_6(A,B)$ are proportional, so their commutator vanishes.
For dimension three take $A=\operatorname{diag}(1,0,-2)$ and
$B=E_{12}+E_{23}$. Direct evaluation gives

$$
[A,B]=E_{12}+2E_{23},\qquad
\operatorname{ad}_A^3(B)=E_{12}+8E_{23},\qquad
J_6(A,B)=-6E_{13}.
$$

Their squared joint Frobenius norm is $7$. Dividing both matrices by
$\sqrt7$ gives a unit tuple and response $-6E_{13}/343$. This also proves
$J_6$ is a nonzero free Lie element. $\square$

Thus response dimension two suffices on the Engel family but fails on general
planar Lie directions. For $p\ge6$, the pure weakly geometric rough path
$\mathbf X_t=\exp_\otimes(tJ_6)$ has identity response in dimension two.
The displayed dimension-three witness has square zero, so its full response
differs from identity by $-6E_{13}/343$. This is full response blindness on
an explicit rough-path class. Evaluation of a truncated group element alone
would not establish full response blindness.

#### 6.2 The three-letter degree-five identity

For three letters define $J_5=[[[e_1,e_2],[e_1,e_3]],e_1]$.
It vanishes on all $2\times2$ tuples. To see this, discard scalar parts and
use the nondegenerate invariant pairing $\langle A,B\rangle=\operatorname{tr}(AB)$
on the three-dimensional space of traceless matrices. Both $[A,B]$ and $[A,C]$
lie in $A^\perp$. Generically they span this plane, and their commutator is
orthogonal to both by trace invariance, hence proportional to $A$. Its bracket
with $A$ vanishes. Polynomiality extends this identity to every tuple.

It is nonzero in the free Lie algebra: for
$A=\operatorname{diag}(1,0,-2)$, $B=E_{12}$ and $C=E_{23}$,
$[[[A,B],[A,C]],A]=-6E_{13}$.
Combined with full sampled ranks through degree four, this verifies
$m_2(d)=5$ for $d\ge3$. The identity is standard polynomial-identity theory;
no novelty is claimed.

### 7. The response gauge is the tangent path discrepancy

This is a uniform first-order matrix-exponential lemma. Its pure-layer
amplitude scaling differs from Carnot homogeneous distances (Le Donne 2016,
Definition 2.1); no standalone geometric novelty is claimed.

The preceding constants are not only properties of isolated tensor levels.
They give the infinitesimal geometry of the exact response discrepancy on pure
rough paths.

Let $L\in\mathcal L_k(\mathbb R^d)$ be homogeneous of degree $k$, and let

$$
\mathbf X_t^\varepsilon=\exp_\otimes(t\varepsilon L),
\qquad 0\le t\le1,
$$

be the corresponding pure level-$k$ weakly geometric $p$-rough path, with
$p\ge k$. Write
$\mathbf 1$ for the constant rough path.  A matrix representation
$\rho_A:e_i\mapsto A_i$ sends tensor-group exponentials to matrix
exponentials, hence

$$
\Phi^A(\mathbf X^\varepsilon)=
\exp\!\left(\varepsilon\rho_A(L)\right),
\qquad
\Phi^A(\mathbf1)=I_n.
$$

#### Theorem

For $r>0$ and $n\ge1$,

$$
\boxed{
\lim_{\varepsilon\to0}
\frac{d_{r,n}(\mathbf X^\varepsilon,\mathbf1)}{|\varepsilon|}
=r^k\gamma_{k,n}(L).
}
$$

More quantitatively, set

$$
C=r^k\lVert L\rVert_{\mathrm{HS}}.
$$

Then for every $\varepsilon\ne0$,

$$
\boxed{
\left|
\frac{d_{r,n}(\mathbf X^\varepsilon,\mathbf1)}{|\varepsilon|}
-r^k\gamma_{k,n}(L)
\right|
\le
\frac{|\varepsilon|}{2}C^2e^{|\varepsilon|C}.
}
$$

*Proof.* Put $M_A=\rho_A(L)$.  Homogeneity of $L$ and the level-wise
Hilbert--Schmidt bound give

$$
\sup_{\lVert A\rVert_F\le r}\lVert M_A\rVert_{\mathrm{op}}
=r^k\gamma_{k,n}(L),
\qquad
\lVert M_A\rVert_{\mathrm{op}}\le C.
$$

The matrix exponential remainder satisfies

$$
\left\lVert
e^{\varepsilon M}-I-\varepsilon M
\right\rVert_{\mathrm{op}}
\le
\frac{|\varepsilon|^2}{2}
\lVert M\rVert_{\mathrm{op}}^2
e^{|\varepsilon|\lVert M\rVert_{\mathrm{op}}}.
$$

After division by $|\varepsilon|$, the reverse triangle inequality shows that
$\lVert e^{\varepsilon M_A}-I\rVert_{\mathrm{op}}/|\varepsilon|$ differs from
$\lVert M_A\rVert_{\mathrm{op}}$ by at most
$|\varepsilon|C^2e^{|\varepsilon|C}/2$, uniformly over the coefficient ball.
For two bounded functions $f,g$ on the same set,
$|\sup f-\sup g|\le\sup|f-g|$.  Taking the two suprema proves the quantitative
bound and therefore the limit. $\square$

#### Exact truncated form

The tensor exponential is

$$
\exp_\otimes(\varepsilon L)
=1+\varepsilon L+\frac{\varepsilon^2}{2}L^{\otimes2}+\cdots,
$$

whose nonconstant terms have degrees $k,2k,3k,\ldots$.  Consequently, for any
truncation depth $N$ with $k\le N<2k$,

$$
\boxed{
d^{(\le N)}_{r,n}(\mathbf X^\varepsilon,\mathbf1)
=|\varepsilon|r^k\gamma_{k,n}(L).
}
$$

Over the same depths, the Hilbert--Schmidt response bound is exactly
$|\varepsilon|r^k\lVert L\rVert_{\mathrm{HS}}$.  Their ratio is therefore
$\rho_{k,n}(L)$ without taking a limit.

#### Compression consequence

Suppose a representation discards the pure component $\varepsilon L$ and
replaces $\mathbf X^\varepsilon$ by $\mathbf1$.  For any constant

$$
\rho_{k,n}(L)<c<1,
$$

choose response tolerance

$$
\eta_\varepsilon
=c|\varepsilon|r^k\lVert L\rVert_{\mathrm{HS}}.
$$

The Hilbert--Schmidt certificate rejects this representation for every
$\varepsilon\ne0$, because its bound is larger than $\eta_\varepsilon$.  The
quantitative theorem shows that the true exact response discrepancy accepts it
for all sufficiently small $|\varepsilon|$.  Thus directional sharpness can
change an actual compression decision, rather than merely improve a numerical
constant.

This can be made explicit.  Put $x=|\varepsilon|C$.  Dividing the quantitative
remainder by the Hilbert--Schmidt certificate gives

$$
\frac{d_{r,n}(\mathbf X^\varepsilon,\mathbf1)}
{|\varepsilon|r^k\lVert L\rVert_{\mathrm{HS}}}
\le
\rho_{k,n}(L)+\frac{x}{2}e^x.
$$

Hence it is sufficient that

$$
x<\min\left\{1,\frac{2(c-\rho_{k,n}(L))}{e}\right\}
$$

for the true response criterion to accept deletion at tolerance
$\eta_\varepsilon$ while the Hilbert--Schmidt certificate rejects it.

For the Engel family $E_k$, the HS/local-response factor $1/\rho_{k,n}(E_k)$
grows exponentially with $k$ by §5. The normalized tolerance interval
$(\rho_{k,n}(E_k),1)$ has width tending to one.

### 8. Composition and structured response classes

#### 8.1 Telescoping bound for interval responses

Let $P_j=P_j^A(X)$ and $Q_j=P_j^A(\widetilde X)$ be truncated response
factors on interval $j$. Chronological products satisfy

$$
P_1\cdots P_m-Q_1\cdots Q_m
=
\sum_{j=1}^m
P_1\cdots P_{j-1}(P_j-Q_j)Q_{j+1}\cdots Q_m.
$$

This follows by adding and subtracting products in which factors are replaced
from left to right. Submultiplicativity gives

$$
\begin{aligned}
\lVert P_1\cdots P_m-Q_1\cdots Q_m\rVert_{\mathrm{op}}
\le
\sum_{j=1}^m
&\left(\prod_{\ell=1}^{j-1}\lVert P_\ell\rVert_{\mathrm{op}}\right)
\lVert P_j-Q_j\rVert_{\mathrm{op}}\\
&\left(\prod_{\ell=j+1}^{m}\lVert Q_\ell\rVert_{\mathrm{op}}\right).
\end{aligned}
$$

For $\lVert A\rVert_F\le r$, level-wise Hilbert--Schmidt bound yields

$$
\lVert P_j^A(X)\rVert_{\mathrm{op}}
\le
1+\sum_{k=1}^L r^k
\lVert S^{(k)}_{t_{j-1},t_j}(X)\rVert_{\mathrm{HS}}
=:M_j(X).
$$

Write $d_j^{(\le L)}$ for common-$A$ truncated response discrepancy on
interval $j$. Taking supremum over coefficient ball gives

$$
\boxed{
\widetilde d_{r,n,L,\pi}(X,\widetilde X)
\le
\sum_{j=1}^m
\left(\prod_{\ell=1}^{j-1} M_\ell(X)\right)
d_j^{(\le L)}
\left(\prod_{\ell=j+1}^{m}M_\ell(\widetilde X)\right).
}
$$

Thus interval errors propagate with explicit response growth factors. If all
factors are bounded by $M$, right side is at most
$M^{m-1}\sum_jd_j^{(\le L)}$. Exact unitary responses and contractive
discretizations have $M\le1$.

#### 8.2 Block-diagonal reduction

For matrix class $\mathcal C$, define constrained homogeneous gauge

$$
\gamma^{\mathcal C}_{k,n}(D)
=
\sup_{\substack{A\in\mathcal C\\\lVert A\rVert_F\le1}}
\lVert R_{A,k}(D)\rVert_{\mathrm{op}}.
$$

Let $\mathrm{BD}(b,n)$ contain tuples whose matrices share consecutive
block-diagonal structure with largest block size $b\le n$, including one block
of size $b$. Then

$$
\boxed{
\gamma^{\mathrm{BD}(b,n)}_{k,n}(D)=\gamma_{k,b}(D).
}
$$

*Proof.* Write $A^{(q)}$ for tuple restricted to block $q$, with
$a_q=\lVert A^{(q)}\rVert_F$. Homogeneity and definition of $\gamma$ give

$$
\lVert R_{A^{(q)},k}(D)\rVert_{\mathrm{op}}
\le a_q^k\gamma_{k,b_q}(D)
\le a_q^k\gamma_{k,b}(D).
$$

Response matrix is block diagonal, so its operator norm is maximum over
blocks. Since $\sum_qa_q^2\le1$, upper bound is $\gamma_{k,b}(D)$. Concentrating
entire norm budget in one size-$b$ block attains corresponding supremum.
$\square$

For $b=1$, all matrices commute, so every free Lie element of degree at least
two has zero response. For $b=2$, §2 shows full dense value is retained on every
Engel direction. Sections 3 and 4 further give full planar values at levels two
and three, and mixed level-four value $\gamma_{4,2}(G_4)$. Whether larger dense
blocks increase mixed level-four value remains open.

#### 8.3 Exact coordinate-plane loop-order response

Let $U$ be a unit square loop in coordinate plane $(e_1,e_2)$ and $V$ a unit
square loop in plane $(e_2,e_3)$, with corresponding degree-two signatures

$$
u_2=[e_1,e_2],
\qquad
v_2=[e_2,e_3].
$$

Compare concatenations $U*V$ and $V*U$. Since both loops have zero increment,
their signatures start at degree two. Chen products in opposite orders first
differ at degree four:

$$
S(U)S(V)-S(V)S(U)
=u_2\otimes v_2-v_2\otimes u_2+\text{terms of degree at least five}.
$$

Consequently their global signatures agree through degree three. Their
depth-two interval factors are

$$
P_U=I+[A_1,A_2],
\qquad
P_V=I+[A_2,A_3].
$$

Composition in opposite orders gives

$$
P_UP_V-P_VP_U
=
[[A_1,A_2],[A_2,A_3]].
$$

For every $n\ge2$,

$$
\boxed{
\widetilde d_{r,n,2,\{U,V\}}(U*V,V*U)
=\frac{r^4}{2\sqrt2}.
}
$$

*Proof.* Put $a=\lVert A_1\rVert_F$, $b=\lVert A_2\rVert_F$ and
$c=\lVert A_3\rVert_F$. Three applications of operator/Frobenius comparison
and Böttcher--Wenzel give

$$
\begin{aligned}
\lVert[[A_1,A_2],[A_2,A_3]]\rVert_{\mathrm{op}}
&\le
\sqrt2\lVert[A_2,A_3]\rVert_F\lVert[A_1,A_2]\rVert_F\\
&\le2\sqrt2\,ab^2c.
\end{aligned}
$$

Under $a^2+b^2+c^2\le r^2$, first use
$ac\le(r^2-b^2)/2$, then maximize $b^2(r^2-b^2)$. This gives upper bound
$r^4/(2\sqrt2)$.

For attainment at $r=1$, set

$$
H=\frac1{\sqrt2}\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\quad
N=\begin{pmatrix}0&1\\0&0\end{pmatrix},
\quad
M=\begin{pmatrix}0&0\\1&0\end{pmatrix},
$$

and

$$
A_1=\frac12H,
\qquad
A_2=\frac1{\sqrt2}N,
\qquad
A_3=\frac12M.
$$

Tuple Frobenius norm is one. Relations
$[H,N]=\sqrt2N$ and $[N,M]=\sqrt2H$ give

$$
[[A_1,A_2],[A_2,A_3]]
=-\frac1{2\sqrt2}N,
$$

whose operator norm is $1/(2\sqrt2)$. Scaling tuple by $r$ gives factor $r^4$;
embedding block proves every $n\ge2$. $\square$

Diagonal responses give zero because their interval factors commute. By §8.2,
every block-diagonal class containing a $2\times2$ block attains same value.

#### 8.4 Exact response-aware merge gap

Keep loops $U,V$ of §8.3. Fine depth-two representation has factors

$$
P_U=I+C_U,
\qquad
P_V=I+C_V,
\qquad
C_U=[A_1,A_2],\quad C_V=[A_2,A_3].
$$

Merged interval has degree-two signature $u_2+v_2$, hence one factor
$P_{U*V}=I+C_U+C_V$. Therefore merge error is

$$
P_UP_V-P_{U*V}=C_UC_V=[A_1,A_2][A_2,A_3].
$$

For unit loops and every $n\ge2$,

$$
\boxed{
\sup_{\lVert A\rVert_F\le r}
\lVert P_UP_V-P_{U*V}\rVert_{\mathrm{op}}
=\frac{r^4}{4}.
}
$$

*Proof.* With $a,b,c$ as in §8.3,

$$
\lVert[A_1,A_2][A_2,A_3]\rVert_{\mathrm{op}}
\le
2ab^2c.
$$

Use $ac\le(r^2-b^2)/2$ and maximize $b^2(r^2-b^2)$ to obtain $r^4/4$.
At $r=1$, use matrices $H,N,M$ of §8.3 and set

$$
A_1=\frac12N,
\qquad
A_2=\frac1{\sqrt2}H,
\qquad
A_3=\frac12M.
$$

Then

$$
[A_1,A_2][A_2,A_3]
=\frac14NM
=\frac14
\begin{pmatrix}1&0\\0&0\end{pmatrix},
$$

which has operator norm $1/4$. Scaling and block embedding complete proof.
$\square$

Fine response polynomial and merged factor first differ at degree four by
$u_2\otimes v_2$. Since $\lVert u_2\rVert_{\mathrm{HS}}=
\lVert v_2\rVert_{\mathrm{HS}}=\sqrt2$, Hilbert--Schmidt certificate is

$$
r^4\lVert u_2\otimes v_2\rVert_{\mathrm{HS}}=2r^4.
$$

Certificate therefore exceeds exact response error by factor eight. If loop
side length is $s$, both values gain factor $s^4$. Ratio remains $1/8$ at every
scale. Two depth-two tensor-signature intervals in dimension three store 24
positive-level coordinates; merged interval stores 12. Any tolerance

$$
\frac{s^4r^4}{4}<\eta<2s^4r^4
$$

accepts this 50% storage reduction under exact response criterion while
Hilbert--Schmidt certificate rejects it.

This theorem concerns the truncated tensor-signature Euler factors defined in
part A, §10. Ferrucci, Perrée and Lyons store truncated
log-signatures and also analyse log-ODE approximations. Their merged log-ODE
factor is not $I+C_U+C_V$, so the exact constants above do not transfer
automatically. The next calculation recovers their leading small-scale form.

#### 8.5 Log-ODE merge gap at small scale

There is nevertheless a sharp log-ODE analogue at tangent scale. Let the two
square loops have side length $s$, let the coefficient bound be $r$, and put

$$
t=s^2r^2.
$$

After writing $A_i=rB_i$, with $\sum_i\lVert B_i\rVert_F^2\le1$, set

$$
X=[B_1,B_2],\qquad Y=[B_2,B_3].
$$

The fine depth-two log-ODE response is $e^{tX}e^{tY}$, whereas merging the two
intervals and truncating their log-signature at degree two gives
$e^{t(X+Y)}$. Define

$$
\delta_n^{\log}(t)
=
\sup_{\sum_i\lVert B_i\rVert_F^2\le1}
\left\lVert e^{tX}e^{tY}-e^{t(X+Y)}\right\rVert_{\mathrm{op}}.
$$

For every $n\ge2$ and $t\ge0$,

$$
\boxed{
\left|
\delta_n^{\log}(t)-\frac{t^2}{4\sqrt2}
\right|
\le \frac{t^3e^t}{3}.
}
$$

In particular,

$$
\boxed{
\delta_n^{\log}(t)
=\frac{t^2}{4\sqrt2}+O(t^3),
\qquad t\downarrow0.
}
$$

*Proof.* Taylor expansion gives, uniformly over the coefficient ball,

$$
e^{tX}e^{tY}-e^{t(X+Y)}
=\frac{t^2}{2}[X,Y]+R_t.
$$

Write $a=\lVert B_1\rVert_F$, $b=\lVert B_2\rVert_F$ and
$c=\lVert B_3\rVert_F$. Böttcher--Wenzel and Cauchy--Schwarz give

$$
\lVert X\rVert_{\mathrm{op}}+\lVert Y\rVert_{\mathrm{op}}
\le \sqrt2b(a+c)
\le 2b\sqrt{1-b^2}
\le1.
$$

The total operator-norm remainder after degree two is therefore bounded by

$$
\lVert R_t\rVert_{\mathrm{op}}
\le
2\sum_{m=3}^{\infty}\frac{t^m}{m!}
\le\frac{t^3e^t}{3}.
$$

Section 8.3 proves

$$
\sup_{\sum_i\lVert B_i\rVert_F^2\le1}
\lVert[X,Y]\rVert_{\mathrm{op}}
=\frac1{2\sqrt2}
$$

for every $n\ge2$. Taking suprema in the uniform expansion proves the bound.
$\square$

At tensor level the leading merge discrepancy is
$\tfrac12[u_2,v_2]$. The two words $u_2\otimes v_2$ and
$v_2\otimes u_2$ are orthogonal and each has Hilbert--Schmidt norm $2$, hence

$$
\left\lVert\tfrac12[u_2,v_2]\right\rVert_{\mathrm{HS}}=\sqrt2.
$$

The leading Hilbert--Schmidt certificate is therefore $\sqrt2t^2$, while the
true leading response is $t^2/(4\sqrt2)$. Their ratio is again exactly $1/8$.
Thus the factor-eight gap survives for log-ODE merging infinitesimally, although
the finite-$t$ response need not equal its leading term.

#### 8.6 Finite-scale log-ODE bounds and compression separation

The log-ODE response of §8.5 admits a nonasymptotic sandwich. For every
$n\ge2$ and $t\ge0$,

$$
\boxed{
\frac{2(e^{t/2}-1)-t}{\sqrt2}
\le
\delta_n^{\log}(t)
\le
\frac{t^2e^t}{4\sqrt2}.
}
$$

*Upper bound.* The error representation is standard: Iserles and Kropielnicka
(arXiv:2401.06635), Theorem 2.1 and §3.1. The application here compares it with
the exact tensor certificate. For fixed matrices $X,Y$, define

$$
F(s)=e^{(t-s)(X+Y)}e^{sX}e^{sY},
\qquad 0\le s\le t.
$$

Then $F(0)=e^{t(X+Y)}$, $F(t)=e^{tX}e^{tY}$, and differentiation gives

$$
F'(s)
=e^{(t-s)(X+Y)}[e^{sX},Y]e^{sY},
$$

with

$$
[e^{sX},Y]
=\int_0^s e^{(s-u)X}[X,Y]e^{uX}\,du.
$$

Put $x=\lVert X\rVert_{\mathrm{op}}$ and
$y=\lVert Y\rVert_{\mathrm{op}}$. Submultiplicativity yields

$$
\lVert F'(s)\rVert_{\mathrm{op}}
\le
s e^{t(x+y)}\lVert[X,Y]\rVert_{\mathrm{op}}.
$$

Section 8.5 gives $x+y\le1$, and §8.3 gives
$\lVert[X,Y]\rVert_{\mathrm{op}}\le1/(2\sqrt2)$. Integrating from zero to
$t$ proves the upper bound.

*Lower bound.* In the matrices $H,S,K$ of §4.2, take the following equivalent
explicit tuple:

$$
B_1=
\begin{pmatrix}
1/4&0\\
-1/(2\sqrt2)&-1/4
\end{pmatrix},
\quad
B_2=
\begin{pmatrix}
0&1/\sqrt2\\
0&0
\end{pmatrix},
\quad
B_3=
\begin{pmatrix}
-1/4&0\\
-1/(2\sqrt2)&1/4
\end{pmatrix}.
$$

Its joint Frobenius norm is one, and

$$
X=[B_1,B_2]
=
\begin{pmatrix}
1/4&1/(2\sqrt2)\\
0&-1/4
\end{pmatrix},
\qquad
Y=[B_2,B_3]
=
\begin{pmatrix}
-1/4&1/(2\sqrt2)\\
0&1/4
\end{pmatrix}.
$$

Direct exponentiation gives

$$
e^{tX}e^{tY}-e^{t(X+Y)}
=
\frac{2(e^{t/2}-1)-t}{\sqrt2}
\begin{pmatrix}0&1\\0&0\end{pmatrix}.
$$

The scalar coefficient is nonnegative. Its operator norm proves the lower
bound for $n=2$, and block embedding proves it for all $n\ge2$. $\square$

The lower bound refines the tangent expansion constructively:

$$
\delta_n^{\log}(t)
\ge
\frac{t^2}{4\sqrt2}
+\frac{t^3}{24\sqrt2}
+O(t^4).
$$

The complete weighted Hilbert--Schmidt certificate is also explicit. Let
$U=u_2$ and $V=v_2$. They are orthogonal degree-two tensors with
$\lVert U\rVert_{\mathrm{HS}}=\lVert V\rVert_{\mathrm{HS}}=\sqrt2$. At
tensor degree $2m$, the fine and merged signatures differ by

$$
D_{2m}
=
\sum_{p+q=m}\frac{U^{\otimes p}\otimes V^{\otimes q}}{p!q!}
-\frac{(U+V)^{\otimes m}}{m!}.
$$

Tensor words in the orthogonal alphabet $\{U,V\}$ are mutually orthogonal.
There are $m+1$ words of the form $U^pV^{m-p}$ and $2^m-m-1$ remaining words.
Using $\sum_{p=0}^m\binom mp^2=\binom{2m}m$ gives

$$
\boxed{
\lVert D_{2m}\rVert_{\mathrm{HS}}
=
\frac{2^{m/2}}{m!}
\sqrt{\binom{2m}{m}-2^m}.
}
$$

Consequently the full certificate for $t=s^2r^2$ is

$$
\boxed{
\mathcal B(t)
=
\sum_{m=2}^{\infty}
t^m\frac{2^{m/2}}{m!}
\sqrt{\binom{2m}{m}-2^m}.
}
$$

Its first two terms are

$$
\mathcal B(t)
\ge
\sqrt2t^2+\frac{2\sqrt6}{3}t^3.
$$

In fact, the certificate has a simple global lower bound. For $m\ge2$,
$\binom{2m}{m}/2^m$ is increasing and equals $3/2$ at $m=2$, because

$$
\frac{\binom{2m+2}{m+1}/2^{m+1}}
{\binom{2m}{m}/2^m}
=\frac{2m+1}{m+1}>1.
$$

Hence

$$
\binom{2m}{m}-2^m\ge2^{m-1},
$$

and termwise summation gives

$$
\boxed{
\mathcal B(t)
\ge
\frac{e^{2t}-1-2t}{\sqrt2}.
}
$$

For every $t>0$,

$$
\frac{t^2e^t}{4\sqrt2}<\mathcal B(t).
$$

Indeed,

$$
e^{2t}-1-2t
=\sum_{k=2}^{\infty}\frac{2^kt^k}{k!},
\qquad
\frac{t^2e^t}{4}
=\sum_{k=2}^{\infty}\frac{t^k}{4(k-2)!}.
$$

The first coefficient strictly exceeds the second at every $k\ge2$, since
$2^{k+2}>k(k-1)$. This inequality is immediate at $k=2,3$ and propagates for
$k\ge3$ because $(k+1)/(k-1)\le2$.

Therefore every tolerance satisfying

$$
\frac{t^2e^t}{4\sqrt2}<\eta<\mathcal B(t),
\qquad t>0,
$$

provably accepts the 50% log-signature storage reduction under the true
response criterion while the Hilbert--Schmidt certificate rejects it. This is
a finite-scale log-ODE compression separation; it does not depend on numerical
optimization or an exact formula for $\delta_n^{\log}(t)$.

#### 8.7 Orthogonal-block merge theorem

The certificate calculation extends from coordinate-plane loops to arbitrary
orthogonal homogeneous blocks. Let $U,V$ belong to the same homogeneous tensor
layer and satisfy

$$
\langle U,V\rangle_{\mathrm{HS}}=0,
\qquad
u=\lVert U\rVert_{\mathrm{HS}},
\qquad
v=\lVert V\rVert_{\mathrm{HS}}.
$$

The tensor identity holds without a Lie assumption. Its interpretation as
log-ODE path compression requires Lie log blocks or the explicitly specified
decoded factors. A merge discrepancy between two approximations must also be
combined with fine approximation error when assessing error to an original path.

Compare the fine tensor exponential
$\exp_\otimes(U)\exp_\otimes(V)$ with the merged one
$\exp_\otimes(U+V)$. At meta-degree $m$ their difference $D_m$ obeys

$$
\boxed{
\lVert D_m\rVert_{\mathrm{HS}}^2
=
\frac1{(m!)^2}
\left[
\sum_{p=0}^m
\binom mp^2 u^{2p}v^{2(m-p)}
-(u^2+v^2)^m
\right].
}
$$

Equivalently, in a manifestly nonnegative form,

$$
\lVert D_m\rVert_{\mathrm{HS}}^2
=
\frac1{(m!)^2}
\sum_{p=1}^{m-1}
\binom mp\left(\binom mp-1\right)
u^{2p}v^{2(m-p)}.
$$

*Proof.* A word containing $p$ copies of $U$ and $m-p$ copies of $V$ has
squared norm $u^{2p}v^{2(m-p)}$. In the fine exponential only the monotone word
$U^pV^{m-p}$ occurs, with coefficient $1/(p!(m-p)!)$; every word has coefficient
$1/m!$ in the merged exponential. Orthogonality of $U,V$ makes distinct words
orthogonal. Summing squared coefficient differences and using the binomial
expansion of $(u^2+v^2)^m$ gives the formula. $\square$

The certificate combines with two response gauges to give a scale-uniform merge
criterion, including unequal block norms. For every admissible unit coefficient
tuple $A$, write

$$
X_A=R_A(U),
\qquad
Y_A=R_A(V),
$$

and define

$$
M=\sup_A
\bigl(\lVert X_A\rVert_{\mathrm{op}}+
\lVert Y_A\rVert_{\mathrm{op}}\bigr),
\qquad
K=\sup_A\lVert[X_A,Y_A]\rVert_{\mathrm{op}}.
$$

At scale $\lambda>0$, let

$$
\delta_{U,V}(\lambda)
=
\sup_A
\left\lVert
e^{\lambda X_A}e^{\lambda Y_A}
-e^{\lambda(X_A+Y_A)}
\right\rVert_{\mathrm{op}}.
$$

The Lie--Trotter argument of §8.6 gives

$$
\boxed{
\delta_{U,V}(\lambda)
\le
\frac{\lambda^2K}{2}e^{\lambda M}.
}
$$

For $m\ge2$, put

$$
H_m(u,v)
=
\sqrt{
\sum_{p=1}^{m-1}
\binom mp\left(\binom mp-1\right)
u^{2p}v^{2(m-p)}
}.
$$

The exact certificate at scale $\lambda$ is

$$
\mathcal B_{U,V}(\lambda)
=
\sum_{m=2}^{\infty}\frac{\lambda^m}{m!}H_m(u,v).
$$

Assume $M>0$ and define

$$
T(u,v;M)
=
\inf_{m\ge2}
\frac{2H_m(u,v)}{m(m-1)M^{m-2}}.
$$

If

$$
\boxed{K<T(u,v;M),}
$$

then

$$
\boxed{
\delta_{U,V}(\lambda)
<\mathcal B_{U,V}(\lambda)
\qquad\text{for every }\lambda>0.
}
$$

Indeed, the coefficient of $\lambda^m$ in the exponential response upper bound
is

$$
\frac K2\frac{M^{m-2}}{(m-2)!},
$$

while the exact certificate coefficient is $H_m(u,v)/m!$. The condition
$K<T(u,v;M)$ makes the former strictly smaller at every $m\ge2$.
If $M=0$, then $X_A=Y_A=0$ for every admissible $A$, so the response vanishes
and separation is immediate whenever $u,v>0$. $\square$

For equal-norm blocks $u=v=a>0$, the exact certificate simplifies to

$$
\mathcal B_{U,V}(\lambda)
=
\sum_{m=2}^{\infty}
\frac{(\lambda a)^m}{m!}
\sqrt{\binom{2m}{m}-2^m},
$$

and §8.6 gives

$$
\mathcal B_{U,V}(\lambda)
\ge
\frac{e^{\sqrt2a\lambda}-1-\sqrt2a\lambda}{\sqrt2}.
$$

Set

$$
\theta=\frac{M}{\sqrt2a},
\qquad
\sigma=\frac{K}{\sqrt2a^2},
$$

and, for $0\le\theta<1$, define the positive threshold

$$
C(\theta)
=
\frac{2}{\displaystyle\sup_{m\ge2}
m(m-1)\theta^{m-2}}.
$$

If

$$
\boxed{
\theta<1
\quad\text{and}\quad
\sigma<C(\theta),
}
$$

then for every $\lambda>0$,

$$
\boxed{
\delta_{U,V}(\lambda)
<
\mathcal B_{U,V}(\lambda).
}
$$

*Proof.* The coefficient of $\lambda^m$ in the exponential response upper bound
remains

$$
\frac K2\frac{M^{m-2}}{(m-2)!},
$$

whereas the coefficient in the displayed certificate lower bound is

$$
\frac{(\sqrt2a)^m}{\sqrt2m!}.
$$

Their ratio is

$$
\frac\sigma2m(m-1)\theta^{m-2}.
$$

The hypothesis makes this ratio strictly below one at every $m\ge2$, so
coefficientwise comparison proves the claim. $\square$

For the coordinate-plane blocks of §§8.5--8.6,

$$
a=\sqrt2,
\qquad M\le1,
\qquad K=\frac1{2\sqrt2}.
$$

Thus $\theta\le1/2$, $\sigma=1/8$, and
$C(1/2)=2/3$, with the supremum in its denominator attained at $m=3,4$.
The general criterion therefore recovers the global separation with substantial
margin.

#### 8.8 Gram-matrix formula for nonorthogonal blocks

Orthogonality is not needed for an exact certificate. Let $U,V$ be arbitrary
elements of the same homogeneous tensor layer and write

$$
a=\lVert U\rVert_{\mathrm{HS}}^2,
\qquad
b=\lVert V\rVert_{\mathrm{HS}}^2,
\qquad
c=\langle U,V\rangle_{\mathrm{HS}}.
$$

Their Gram matrix is positive semidefinite, so $c^2\le ab$. At meta-degree
$m$, define

$$
F_m
=
\sum_{p=0}^m\frac{U^{\otimes p}\otimes V^{\otimes(m-p)}}{p!(m-p)!},
\qquad
G_m
=
\frac{(U+V)^{\otimes m}}{m!}.
$$

Then $D_m=F_m-G_m$ and

$$
\boxed{
\lVert D_m\rVert_{\mathrm{HS}}^2
=
\frac1{(m!)^2}
\left[
\sum_{p,q=0}^m
\binom mp\binom mq
a^{\min(p,q)}
b^{m-\max(p,q)}
c^{|p-q|}
-(a+b+2c)^m
\right].
}
$$

*Proof.* For monotone words
$W_p=U^{\otimes p}\otimes V^{\otimes(m-p)}$,

$$
\langle W_p,W_q\rangle
=
a^{\min(p,q)}b^{m-\max(p,q)}c^{|p-q|}.
$$

This gives the double sum for $\lVert F_m\rVert^2$. Also

$$
\langle W_p,(U+V)^{\otimes m}\rangle
=(a+c)^p(b+c)^{m-p}.
$$

The binomial theorem therefore gives

$$
\langle F_m,G_m\rangle
=\frac{(a+b+2c)^m}{(m!)^2}
=\lVert G_m\rVert^2.
$$

Thus $\lVert F_m-G_m\rVert^2=\lVert F_m\rVert^2-\lVert G_m\rVert^2$, proving
the formula. $\square$

Let

$$
H_m(U,V)=m!\lVert D_m\rVert_{\mathrm{HS}}.
$$

The exact certificate is

$$
\mathcal B_{U,V}(\lambda)
=\sum_{m=2}^{\infty}\frac{\lambda^m}{m!}H_m(U,V).
$$

The response gauges $M,K$ and Lie--Trotter upper bound of §8.7 require no
orthogonality. Hence, for $M>0$, the condition

$$
\boxed{
K<\inf_{m\ge2}
\frac{2H_m(U,V)}{m(m-1)M^{m-2}}
}
$$

implies

$$
\delta_{U,V}(\lambda)<\mathcal B_{U,V}(\lambda)
\qquad\text{for every }\lambda>0.
$$

This is the general two-block merge theorem: the feature-side calculation uses
only the Gram matrix $(a,b,c)$, while the response side uses only $M$ and $K$.
If $U,V$ are collinear, every $H_m$ and $K$ vanish, reflecting that the fine and
merged exponentials agree exactly.

#### 8.9 Arbitrary finite block families

The preceding result extends to merging any finite ordered family, which is
the form needed by an adaptive partition. Let $U_1,\ldots,U_\ell$ belong to the
same homogeneous tensor layer and let

$$
g_{ij}=\langle U_i,U_j\rangle_{\mathrm{HS}}
$$

be their Gram matrix. For a weak composition
$\alpha=(\alpha_1,\ldots,\alpha_\ell)$ of $m$, write
$\alpha!=\prod_i\alpha_i!$, $\binom m\alpha=m!/\alpha!$, and let $w(\alpha)$
be the nondecreasing word with $\alpha_i$ copies of $i$. Define

$$
W_\alpha
=U_1^{\otimes\alpha_1}\otimes\cdots\otimes
U_\ell^{\otimes\alpha_\ell}.
$$

The meta-degree-$m$ parts of the fine and merged representations are

$$
F_m=\sum_{|\alpha|=m}\frac{W_\alpha}{\alpha!},
\qquad
G_m=\frac{(U_1+\cdots+U_\ell)^{\otimes m}}{m!}.
$$

Set $D_m=F_m-G_m$ and $H_m=m!\lVert D_m\rVert_{\mathrm{HS}}$. Then

$$
\boxed{
H_m^2
=
\sum_{|\alpha|=|\beta|=m}
\binom m\alpha\binom m\beta
\prod_{r=1}^m
g_{w(\alpha)_r,w(\beta)_r}
-\left(\sum_{i,j=1}^\ell g_{ij}\right)^m.
}
$$

Thus every level of the exact Hilbert--Schmidt merge certificate depends only
on the $\ell\times\ell$ Gram matrix, even when blocks are nonorthogonal.

*Proof.* Coordinatewise tensor inner products give

$$
\langle W_\alpha,W_\beta\rangle
=\prod_{r=1}^m g_{w(\alpha)_r,w(\beta)_r},
$$

which produces the double sum for $\lVert F_m\rVert^2$. Put
$s_i=\sum_jg_{ij}$. Then

$$
\langle W_\alpha,(U_1+\cdots+U_\ell)^{\otimes m}\rangle
=\prod_i s_i^{\alpha_i}.
$$

The multinomial theorem yields

$$
\langle F_m,G_m\rangle
=\frac1{m!}\sum_{|\alpha|=m}\frac{\prod_i s_i^{\alpha_i}}{\alpha!}
=\frac{(\sum_i s_i)^m}{(m!)^2}
=\lVert G_m\rVert^2.
$$

Therefore $\lVert D_m\rVert^2=\lVert F_m\rVert^2-\lVert G_m\rVert^2$,
which is the displayed formula after multiplication by $(m!)^2$. $\square$

There is also a response-side bound with no dependence on the number of
blocks other than through two aggregate gauges. For fixed $A$, write
$X_i=R_A(U_i)$, and define

$$
M=\sup_A\sum_{i=1}^\ell\lVert X_i\rVert_{\mathrm{op}},
\qquad
K=\sup_A\sum_{1\le i<j\le\ell}
\lVert[X_i,X_j]\rVert_{\mathrm{op}}.
$$

Then

$$
\boxed{
\delta_{U_1,\ldots,U_\ell}(\lambda)
\le\frac{\lambda^2K}{2}e^{\lambda M}.
}
$$

*Proof.* The two-factor Lie--Trotter bound and induction give, for arbitrary
matrices $X_1,\ldots,X_\ell$,

$$
\left\lVert
e^{\lambda X_1}\cdots e^{\lambda X_\ell}
-e^{\lambda\sum_iX_i}
\right\rVert_{\mathrm{op}}
\le
\frac{\lambda^2}{2}
e^{\lambda\sum_i\lVert X_i\rVert_{\mathrm{op}}}
\sum_{i<j}\lVert[X_i,X_j]\rVert_{\mathrm{op}}.
$$

Indeed, append $e^{\lambda X_j}$ to the first $j-1$ factors. The previous
error is multiplied by at most $e^{\lambda\lVert X_j\rVert}$, while the new
two-factor error contains
$[\sum_{i<j}X_i,X_j]=\sum_{i<j}[X_i,X_j]$. Every term is bounded by the same
final exponential factor. Taking the supremum over $A$ proves the claim.
$\square$

Consequently, for $M>0$, the completely explicit coefficientwise condition

$$
\boxed{
K<\inf_{m\ge2}
\frac{2H_m}{m(m-1)M^{m-2}}
}
$$

implies

$$
\delta_{U_1,\ldots,U_\ell}(\lambda)
<
\sum_{m=2}^{\infty}\frac{\lambda^m}{m!}H_m
$$

for every $\lambda>0$. The right side is the exact full weighted
Hilbert--Schmidt certificate. This reduces a many-interval compression decision
to a Gram-matrix calculation and bounds on two response gauges. The condition
is sufficient rather than necessary: summing pairwise commutator norms discards
cancellation between different pairs.

#### 8.10 Certified coordinate-loop chains

The multiblock criterion is nonvacuous in a natural path example. In driver
dimension four, take the three adjacent coordinate-plane area tensors

$$
U_1=[e_1,e_2],
\qquad
U_2=[e_2,e_3],
\qquad
U_3=[e_3,e_4].
$$

They are pairwise orthogonal and each has Hilbert--Schmidt norm $\sqrt2$.
For $X_i=R_A(U_i)=[A_i,A_{i+1}]$, put
$a_i=\lVert A_i\rVert_F$ and

$$
c_*=\cos\frac\pi5.
$$

The largest eigenvalue of the four-vertex path quadratic form gives

$$
\sum_{i=1}^3a_ia_{i+1}
\le c_*\sum_{i=1}^4a_i^2
\le c_*.
$$

Böttcher--Wenzel therefore yields the response-gauge bounds

$$
M
\le\sqrt2\sum_{i=1}^3a_ia_{i+1}
\le\sqrt2c_*,
$$

and

$$
\begin{aligned}
K
&\le\sqrt2\sum_{i<j}\lVert X_i\rVert_F\lVert X_j\rVert_F\\
&\le2\sqrt2\sum_{i<j}
(a_ia_{i+1})(a_ja_{j+1})\\
&\le\sqrt2\left(\sum_{i=1}^3a_ia_{i+1}\right)^2
\le\sqrt2c_*^2.
\end{aligned}
$$

It remains to lower-bound the exact certificate coefficients. After
multiplication by $m!$, every nonmonotone word in the orthogonal alphabet
$\{U_1,U_2,U_3\}$ occurs in the merged representation with coefficient one
and is absent from the fine representation. There are $3^m$ words in total and
$\binom{m+2}{2}$ nondecreasing words. Hence

$$
H_m^2
\ge2^m\left(3^m-\binom{m+2}{2}\right)
\ge2^m3^{m-1},
\qquad m\ge2,
$$

where the last inequality follows from
$\binom{m+2}{2}\le2\cdot3^{m-1}$.
Using the upper bound $M\le\sqrt2c_*$,

$$
\frac{2H_m}{m(m-1)M^{m-2}}
\ge
L_m
:=
\frac{4\,3^{(m-1)/2}c_*^{-(m-2)}}{m(m-1)}.
$$

Now $L_2=2\sqrt3$, $L_3=2/c_*$, and

$$
\frac{L_{m+1}}{L_m}
=\frac{\sqrt3}{c_*}\frac{m-1}{m+1}>1,
\qquad m\ge3.
$$

Thus every coefficient threshold is at least $2/c_*$, whereas

$$
K\le\sqrt2c_*^2<\frac2{c_*}.
$$

The theorem of §8.9 now gives

$$
\boxed{
\delta_{U_1,U_2,U_3}(\lambda)
<\mathcal B_{U_1,U_2,U_3}(\lambda)
\quad\text{for every }\lambda>0.
}
$$

Hence merging these three depth-two log-signature intervals into one is
accepted by a nonempty interval of response tolerances that the exact weighted
Hilbert--Schmidt certificate rejects, at every positive scale. In a generic
dimension-four depth-two log-signature layout, this changes storage from three
copies of ten coordinates to one copy of ten.

The same argument gives an infinite family. For any $\ell\ge3$, work in driver
dimension $\ell+1$ and take

$$
U_i=[e_i,e_{i+1}],
\qquad 1\le i\le\ell.
$$

Put $c_\ell=\cos(\pi/(\ell+2))$. The path-graph quadratic form and the same two
Böttcher--Wenzel estimates give

$$
M\le\sqrt2c_\ell<\sqrt2,
\qquad
K\le\sqrt2c_\ell^2<\sqrt2.
$$

On the feature side, retain only nonmonotone words over
$\{U_1,U_2,U_3\}$. Their contributions are orthogonal to every other word, so
the lower bound

$$
H_m^2\ge2^m3^{m-1}
$$

continues to hold independently of $\ell$. Consequently,

$$
\frac{2H_m}{m(m-1)M^{m-2}}
\ge
\widetilde L_m
:=
\frac{4\,3^{(m-1)/2}}{m(m-1)}.
$$

The ratio

$$
\frac{\widetilde L_{m+1}}{\widetilde L_m}
=\sqrt3\,\frac{m-1}{m+1}
$$

is below one through $m=3$ and above one from $m=4$ onward. Therefore the
minimum occurs at $m=4$ and equals
$\widetilde L_4=\sqrt3$. Since

$$
K<\sqrt2<\sqrt3
\le\inf_{m\ge2}\widetilde L_m,
$$

the coefficientwise condition holds for every chain length. Thus

$$
\boxed{
\delta_{U_1,\ldots,U_\ell}(\lambda)
<\mathcal B_{U_1,\ldots,U_\ell}(\lambda)
\quad
(\ell\ge3,\ \lambda>0).
}
$$

The bounds sum to a closed-form quantitative version:

$$
\boxed{
\delta_{U_1,\ldots,U_\ell}(\lambda)
<
\frac{\lambda^2}{\sqrt2}e^{\sqrt2\lambda}
<
\frac{e^{\sqrt6\lambda}-1-\sqrt6\lambda}{\sqrt3}
\le
\mathcal B_{U_1,\ldots,U_\ell}(\lambda).
}
$$

The first inequality uses $M,K<\sqrt2$. The last follows by summing
$H_m\ge(\sqrt6)^m/\sqrt3$. The middle inequality is coefficientwise and is
equivalent to $\sqrt2<\widetilde L_m$ for every $m\ge2$.

Merging $\ell$ such intervals into one reduces generic depth-two
log-signature storage by the factor $\ell$, while retaining a nonempty interval
of response tolerances rejected by the exact Hilbert--Schmidt certificate. The
separation margin is uniform in the chain length.

#### 8.11 Spectral response-gauge bounds for degree-two blocks

The abstract gauges $M$ and $K$ admit inexpensive certified bounds for
degree-two Lie blocks. Represent such a block by a skew-symmetric matrix
$\Omega\in\mathbb R^{d\times d}$:

$$
U_\Omega=\sum_{i,j=1}^d\Omega_{ij}e_i\otimes e_j,
\qquad
R_A(U_\Omega)=\sum_{i<j}\Omega_{ij}[A_i,A_j].
$$

First, the individual response gauge is exact.

##### Proposition

For every response dimension $n\ge2$,

$$
\boxed{
\sup_{\lVert A\rVert_F\le1}
\lVert R_A(U_\Omega)\rVert_{\mathrm{op}}
=
\sup_{\lVert A\rVert_F\le1}
\lVert R_A(U_\Omega)\rVert_F
=\frac{\lVert\Omega\rVert_{\mathrm{op}}}{\sqrt2}.
}
$$

*Proof.* An orthogonal driver-coordinate change puts $\Omega$ into real skew
normal form with $2\times2$ blocks
$\sigma_j\left(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\right)$.
Writing $b_k=\lVert B_k\rVert_F$ for the transformed response tuple,
Böttcher--Wenzel gives

$$
\lVert R_A(U_\Omega)\rVert_F
\le\sqrt2\sum_j\sigma_jb_{2j-1}b_{2j}
\le\frac{\sigma_1}{\sqrt2}\sum_kb_k^2
\le\frac{\lVert\Omega\rVert_{\mathrm{op}}}{\sqrt2}.
$$

Concentrate the coefficient budget in a top singular plane and use

$$
B=\begin{pmatrix}1/2&0\\0&-1/2\end{pmatrix},
\qquad
C=\begin{pmatrix}0&1/\sqrt2\\0&0\end{pmatrix}.
$$

Then $\lVert B\rVert_F^2+\lVert C\rVert_F^2=1$ and
$[B,C]=C$, whose operator and Frobenius norms both equal $1/\sqrt2$.
This attains both upper bounds. $\square$

Now let $\Omega^{(1)},\ldots,\Omega^{(\ell)}$ represent the blocks to merge.
Define

$$
W_{ij}=\sum_{s=1}^\ell|\Omega^{(s)}_{ij}|,
\qquad
\Lambda=\lambda_{\max}(W),
\qquad
\sigma_s=\lVert\Omega^{(s)}\rVert_{\mathrm{op}}.
$$

The matrix $W$ is a nonnegative weighted driver-overlap graph. Set

$$
\widehat M
=\frac1{\sqrt2}\min\left\{\Lambda,\sum_s\sigma_s\right\},
$$

and

$$
\widehat K
=\frac1{\sqrt2}\min\left\{
\frac{\Lambda^2}{2},
\sum_{s<t}\sigma_s\sigma_t
\right\}.
$$

##### Theorem

The multiblock response gauges of §8.9 satisfy

$$
\boxed{
M\le\widehat M,
\qquad
K\le\widehat K.
}
$$

*Proof.* Put $a_i=\lVert A_i\rVert_F$. For
$X_s=R_A(U_{\Omega^{(s)}})$, Böttcher--Wenzel and the triangle inequality give

$$
\lVert X_s\rVert_F
\le\sqrt2\sum_{i<j}|\Omega^{(s)}_{ij}|a_ia_j.
$$

Summing in $s$,

$$
\sum_s\lVert X_s\rVert_{\mathrm{op}}
\le\frac1{\sqrt2}a^\mathsf TWa
\le\frac{\Lambda}{\sqrt2}.
$$

The individual proposition also bounds this sum by
$\sum_s\sigma_s/\sqrt2$, proving the claim for $M$.

Write
$q_s=\sum_{i<j}|\Omega^{(s)}_{ij}|a_ia_j$. A second
Böttcher--Wenzel application gives

$$
\begin{aligned}
\sum_{s<t}\lVert[X_s,X_t]\rVert_{\mathrm{op}}
&\le\sqrt2\sum_{s<t}\lVert X_s\rVert_F\lVert X_t\rVert_F\\
&\le2\sqrt2\sum_{s<t}q_sq_t\\
&\le\sqrt2\left(\sum_sq_s\right)^2\\
&\le\frac{\Lambda^2}{2\sqrt2}.
\end{aligned}
$$

Alternatively, the individual bound
$\lVert X_s\rVert_F\le\sigma_s/\sqrt2$ gives
$K\le\sum_{s<t}\sigma_s\sigma_t/\sqrt2$. Taking the better of the two proves
the result. $\square$

Let $G_{st}=\langle\Omega^{(s)},\Omega^{(t)}\rangle_F$. This is exactly the
block Gram matrix required by §8.9, so each $H_m$ is feature-side computable.
For $\widehat M>0$, the sufficient condition

$$
\boxed{
\widehat K
<
\inf_{m\ge2}
\frac{2H_m(G)}{m(m-1)\widehat M^{m-2}}
}
$$

therefore certifies all-scale response separation without optimizing over a
matrix response system. The infimum ranges over infinitely many levels; a
finite scan alone verifies no all-scale claim. The chain proof supplies an
analytic lower bound for that infimum. At one fixed scale, a certificate partial
sum is a valid lower bound and can certify separation from a response upper
bound. Constructing $G$ and $W$ costs
$O(\ell^2d^2+\ell d^2)$, followed by one $d\times d$ symmetric eigenvalue
calculation. For the coordinate-loop chain, $W$ is the path adjacency matrix,
so these bounds recover exactly
$M\le\sqrt2\cos(\pi/(\ell+2))$ and
$K\le\sqrt2\cos^2(\pi/(\ell+2))$ from §8.10.


<a id="storage"></a>
## C. Euler minimax and storage

Euler decoding and dense generic log-signature storage follow FPL,
(1.8)--(1.10); they are distinct from one-factor log-ODE decoding.

### 1. Exact bounded variation minimax error

Write $P_N(x)=\sum_{j=0}^N x^j/j!$. For a bounded variation path $X$, a
coefficient tuple $A$ and a partition with $m$ intervals, put

$$
x_i=\lVert A\rVert_F\lVert X\rVert_{1\text{-var};[t_{i-1},t_i]},
\qquad R=\sum_i x_i.
$$

#### Theorem

For arbitrary interval lengths,

$$
\lVert\Phi^A(X)-E^A_{N,\pi}(X)\rVert_{\mathrm{op}}
\le e^R-\prod_{i=1}^m P_N(x_i).
$$

For $N,m\ge1$, the exact worst error over paths with
$\lVert A\rVert_F\lVert X\rVert_{1\text{-var}}\le r$, allowing a partition
depending on $X$ and independent of $A$, is

$$
\boxed{\mathcal E(N,m,r)=e^r-P_N(r/m)^m.}
$$

An equal variation partition attains the upper bound. A straight line with a
rank-one scalar development attains the worst error for every response
dimension. Equal variation also minimizes the displayed majorant among
partitions with fixed total variation and interval count.

*Proof.* Expand each exact interval factor into its homogeneous response
series. Its degree $j$ term has operator norm at most $x_i^j/j!$, by
signature factorial decay and FPL (1.9). Expand the product of all exact
factors. The Euler product retains precisely terms with every interval degree
at most $N$. Thus the difference is a sum over excluded multi-indices; absolute
convergence and the triangle inequality give

$$
\sum_{\substack{j_1,\ldots,j_m\ge0\\\max j_i>N}}
\prod_i\frac{x_i^{j_i}}{j_i!}
=e^R-\prod_iP_N(x_i).
$$

This also shows the majorant increases in each $x_i$. To minimize it at fixed
$R$, maximize $\sum_i\log P_N(x_i)$. For $N\ge2$,

$$
P_{N-1}(x)^2-P_N(x)P_{N-2}(x)
=\frac{x^{N-1}}{N!}\sum_{j=0}^{N-1}(N-j)\frac{x^j}{j!}\ge0.
$$

Consequently $\log P_N$ is concave on $[0,\infty)$, strictly concave for
$x>0$. For $N=1$ it equals $\log(1+x)$ and is strictly concave. Jensen gives
$\prod_iP_N(x_i)\le P_N(R/m)^m$.

Choose $X_t=t e_1$ on $[0,1]$, $A_1=rQ$ where $Q$ is an orthogonal rank-one
projection, and $A_i=0$ for $i>1$. The exact response on range $Q$ is $e^r$;
on any partition Euler response is $\prod_iP_N(r\Delta t_i)$. Jensen shows
even the best partition has error at least $e^r-P_N(r/m)^m$. Outside range
$Q$ both responses are identity. This proves the minimax equality, including
the permission to choose partitions adaptively. $\square$

FPL already display this scalar expression as a sharpness example in (1.13).
The extension here is its universal upper bound, partition minimization and
resulting exact discrete constraint. Novelty relative to wider numerical
analysis literature remains unverified.

### 2. Exact discrete optimization and asymptotics

Two sources of conservatism are separate. The exact weighted HS certificate
is the sum of homogeneous error norms; FPL's length envelope applies further
tail/product bounds. Straight lines attain the exact scalar error and its HS
certificate. The BV refinement below improves the length envelope, while the
pure Lie comparison in §3 exhibits a response/HS gap within the specified class.

With Witt dimension

$$
\Lambda_d(N)=\sum_{j=1}^N\frac1j\sum_{h\mid j}\mu(h)d^{j/h},
$$

the bounded variation problem becomes the computable integer optimization

$$
\min_{N,m\in\mathbb N_+}m\Lambda_d(N)
\quad\text{subject to}\quad\mathcal E(N,m,r)\le\varepsilon.
$$

For each $N$, $\mathcal E$ decreases in $m$, by concavity of $\log P_N$ and
$P_N(0)=1$. Bisection gives the smallest feasible $m$. Enumeration stops when
$\Lambda_d(N)$ exceeds the best total storage already found: even $m=1$
then costs more, and $\Lambda_d$ increases. This certifies a global integer
minimum without imposing an arbitrary search cap. Log arithmetic locates a
candidate interval count; close feasibility comparisons are refined with
decimal precision exceeding its number of digits. Arithmetic is numerical,
rather than interval-certified.

#### Proposition

Let $C_{d,k}(z,\varepsilon)$ minimize $m\Lambda_d(kq)$ subject to
$\mathcal E(q,m,z)\le\varepsilon$, with integers $q,m\ge1$ and fixed
$d\ge2$, $k\ge1$. Define $\beta=k\log d$ and
$\lambda=z+\log(z/\varepsilon)$. In either limit $z\to\infty$ with
$\varepsilon$ fixed or $\varepsilon\downarrow0$ with $z>0$ fixed,

$$
q_*\sim\sqrt{\lambda/\beta},\qquad
\log(m_*/z)\sim\sqrt{\beta\lambda},\qquad
\log(C_{d,k}/z)\sim2\sqrt{\beta\lambda}.
$$

Replacing $\mathcal E(q,m,z)$ by FPL's bound

$$
B(q,m,z)=\frac{e^z z^{q+1}}{(q+1)!m^q}
$$

changes optimal storage by a relative factor tending to one. Integer optima
can differ at finite parameters. The leading logarithmic formulas above
avoid claiming continuous prefactors for integer optimizers.

*Proof.* Write $x=z/m$ and
$a=e^{-x}\sum_{j=q+1}^\infty x^j/j!$. Then

$$
\mathcal E=e^z[1-(1-a)^m],\qquad
e^{-x}\frac{x^{q+1}}{(q+1)!}\le a
\le\frac{x^{q+1}}{(q+1)!}.
$$

Bernoulli's inequality applied to $(1-a)^{-m}$ yields
$1-(1-a)^m\ge ma/(1+ma)$. Hence

$$
\frac{e^{-x}}{1+ma}\le\frac{\mathcal E}{B}\le1.
$$

If $\mathcal E\le\varepsilon<e^z$, the inequality
$1-(1-a)^m\ge1-e^{-ma}$ gives
$ma\le-\log(1-\varepsilon e^{-z})$.

Taking $q$ nearest $\sqrt{\lambda/\beta}$ and the least integer satisfying
$B\le\varepsilon$ gives
$\log(C_{d,k}/z)\le2\sqrt{\beta\lambda}+O(\log\lambda)$, by Stirling and
FPL Theorem 1.2. In particular any optimizer has $q=O(\sqrt\lambda)$.
Its interval count diverges and $x\to0$: for fixed $z$, a bounded $m$ would
give $\log\mathcal E\ge-O(\sqrt\lambda\log\lambda)$ from the first tail
term, contradicting $\log\varepsilon\sim-\lambda$. For $z\to\infty$, if
$m=1$ then $e^{-z}P_q(z)\to0$ and $\mathcal E\sim e^z$; if $m\ge2$ and
$x\ge c>0$, the first tail term gives
$\log\mathcal E\ge z/2-O(\sqrt z\log z)$, also infeasible. For fixed $z$,
$ma\le z x^q/(q+1)!\le zx/2\to0$ once $x<1$, since $q\ge1$.
For $z\to\infty$ with fixed $\varepsilon$, feasibility gives $ma\to0$
by the displayed logarithmic bound. Thus
$\mathcal E/B\to1$ at every optimizer.

The lower bound above implies
$\log B\le\log\varepsilon+o(1)$. Combining with Witt asymptotics and
Stirling gives, uniformly for optimizing orders,

$$
\log(C_{d,k}/z)
\ge\beta q+\lambda/q-O(\log\lambda).
$$

The minimum of $\beta q+\lambda/q$ is $2\sqrt{\beta\lambda}$; the upper
bound forces $q/\sqrt{\lambda/\beta}\to1$. The formula for $m$ follows by
subtracting $\log\Lambda_d(kq)$ from total storage. Finally inflate $m$ at
an exact optimizer by factor $\max(1,(B/\varepsilon)^{1/q})$, rounding up.
This makes $B\le\varepsilon$ at relative storage cost $1+o(1)$, because
$B/\varepsilon\le1+o(1)$ and $m\to\infty$. Exact and bound constrained
minimal costs therefore have ratio tending to one. $\square$

For $k=1$ this preserves FPL's leading bounded variation asymptotics. Their
universal admissible class contains straight lines, so within-level Lie
response gaps cannot improve that minimax rate.

### 3. Pure Engel paths: rigorous structured storage bounds

Let $E_k=\operatorname{ad}_{e_1}^{k-1}(e_2)$, $k\ge2$, and consider
$\mathbf X_t=\exp_\otimes(t a E_k)$ for $a>0$, a pure degree-$k$ weakly
geometric rough path. Absorb coefficient radius into $a$ by homogeneity:
original amplitude $b$ and radius $r$ give $a=br^k$. Set

$$
h_k=\lVert E_k\rVert_{\mathrm{HS}}
=\sqrt{\binom{2k-2}{k-1}},\qquad
g_k=[2(k-1)]^{(k-1)/2}k^{-k/2},\qquad s_k=g_k/\sqrt2.
$$

For truncation $N=kq$, an equal partition gives Euler response
$P_q(aR_A(E_k)/m)^m$ and exact response $e^{aR_A(E_k)}$, for unit
Frobenius coefficient tuples. Let $\delta_k(q,m,a)$ be their response
supremum, in any response dimension at least two.

#### Theorem

$$
\boxed{
\mathcal E(q,m,a s_k)
\le\delta_k(q,m,a)
\le\mathcal E(q,m,a g_k)
<\mathcal E(q,m,a h_k).
}
$$

The last expression is the exact full weighted Hilbert--Schmidt certificate
for this Euler error. The lower bound holds for arbitrary partitions after
minimizing the attained scalar error over those partitions.

*Proof.* Section 2 of `technical.md` gives
$\lVert R_A(E_k)\rVert_{\mathrm{op}}\le g_k$ uniformly over response
dimension. Expand $e^{aC}-P_q(aC/m)^m$: coefficients of $C^j$ are
nonnegative, since the polynomial product retains a subset of the exponential
product terms. Bounding each $\lVert C^j\rVert\le g_k^j$ gives the upper
bound.

For the lower bound put $\alpha=\sqrt{(k-1)/k}$,
$\eta=1/\sqrt k$ and take

$$
A_1=\frac\alpha{\sqrt2}
\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
A_2=\frac\eta{\sqrt2}
\begin{pmatrix}0&1\\(-1)^{k-1}&0\end{pmatrix}.
$$

Their tuple norm is one, and $R_A(E_k)$ is symmetric with off-diagonal
entries $g_k/\sqrt2$. A unit positive eigenvector gives precisely the scalar
Euler error with scale $a s_k$. Embed this block in larger response dimensions.
The partition lower bound follows from §1's concavity argument.

At tensor degree $kj$, the Euler error is $c_j a^j E_k^{\otimes j}$ with
$c_j\ge0$. Its Hilbert--Schmidt norm is $c_j(a h_k)^j$. Summing gives
$\mathcal E(q,m,a h_k)$ exactly. Since $g_k<h_k$ and $\mathcal E$ is
strictly increasing in its positive scale, the final inequality is strict.
$\square$

Let $C_k^{\mathrm{true}}$ minimize dense generic storage using true response
error and allowing adaptive partitions. For
$\varepsilon<e^{a s_k}-1$, depths below $k$ are infeasible. Intermediate
depths between $kq$ and $k(q+1)$ yield the same Euler polynomial as $kq$, at
larger storage. Thus

$$
\boxed{
C_{d,k}(a s_k,\varepsilon)
\le C_k^{\mathrm{true}}(a,\varepsilon)
\le C_{d,k}(a g_k,\varepsilon)
\le C_{d,k}(a h_k,\varepsilon).
}
$$

The middle computation is a proved feasible representation; its optimum is
an upper bound on the true optimum. An optimum computed from the witness is
a lower bound on true storage, and need not itself be feasible.

#### Corollary: growing storage separation

For fixed $k,d,\varepsilon$ and $a\to\infty$,

$$
\liminf_{a\to\infty}\frac1{\sqrt a}
\log\frac{C_{d,k}(a h_k,\varepsilon)}{C_k^{\mathrm{true}}(a,\varepsilon)}
\ge2\sqrt{k\log d}\,(\sqrt{h_k}-\sqrt{g_k})>0.
$$

This follows directly from §2 and the storage sandwich. Thus the loss from
the HS certificate grows at least stretched exponentially with pure-path
amplitude, within FPL's Euler decoding and generic storage convention. For
fixed $a$ and $\varepsilon\downarrow0$, both ends of the sandwich have
leading log storage $2\sqrt{k\log d\log(1/\varepsilon)}$, so that leading
rate is unchanged.

Scope: a single depth-$k$ log-ODE factor represents this pure path exactly.
A decoder exploiting that fact, or storing only its known nonzero Lie
direction, has a different storage problem. These pure paths are an analytic
Euler benchmark; practical compression must extend to changing, noncommuting
Lie directions or mixed degrees. FPL's general rough-path bound in Theorem
2.8 is a separate constraint; the structured comparison here uses the exact
HS certificate of each pure path rather than that general bound.


<a id="planar-bv"></a>
## D. Planar BV and mixed displacement/area distortion

The sharp planar cubic gauge is proved in part B, §3. The statements below
combine it with uniform tails; executable anchors are in
[the examples notebook](notebooks/examples.ipynb).

### 1. A first-discrepancy lemma

Let $F=1+\sum_{j\ge1}F_j$ and $G=1+\sum_{j\ge1}G_j$ be group-like tensor
series over $\mathbb R^2$, with homogeneous components and absolutely
convergent norm majorants after dilation $\delta_t$, $t\ge0$.
Assume $F_1=G_1$ and $F_2=G_2$. Then
$D_3=F_3-G_3$ is a degree-three free Lie element: it equals
$\log_3 F-\log_3 G$, because all lower terms in the formal logarithm cancel.

Put $h=\lVert D_3\rVert_{\mathrm{HS}}$,
$\rho=2\sqrt2/9$, and define

$$
\Delta_n(t)=\sup_{\lVert A\rVert_F\le1}
\lVert R_A(\delta_t F)-R_A(\delta_t G)\rVert_{\mathrm{op}},
\qquad
\mathcal B(t)=\sum_{j\ge3}t^j\lVert F_j-G_j\rVert_{\mathrm{HS}}.
$$

Here $R_A$ evaluates full tensor series by chronological matrix products.
Suppose response and HS tails satisfy, uniformly over the coefficient ball,

$$
\left\lVert\sum_{j\ge4}t^jR_A(F_j-G_j)\right\rVert_{\mathrm{op}}
\le Q_R(t),
\qquad
\sum_{j\ge4}t^j\lVert F_j-G_j\rVert_{\mathrm{HS}}\le Q_H(t).
$$

For every $n\ge2$, `technical.md` §3 gives
$\gamma_{3,n}(D_3)=\rho h$. Reverse triangle inequality and uniformity of
the remainder therefore imply

$$
\boxed{
|\Delta_n(t)-\rho h t^3|\le Q_R(t),
\qquad
h t^3\le\mathcal B(t)\le h t^3+Q_H(t).
}
$$

If $h>0$ and both tails are $O(t^4)$, then
$\Delta_n(t)/\mathcal B(t)\to\rho$, uniformly in $n\ge2$.
Quantitatively,
$|\Delta_n(t)/\mathcal B(t)-\rho|
\le(Q_R(t)+\rho Q_H(t))/(ht^3)$.
This is the full response of the displayed series; no evaluation of a finite
nilpotent quotient is used.



### 2. Positive tail and explicit separation radius

For $a,b\ge0$, define the degree-four-and-above tail

$$
T_4(a,b;t)
=e^{at+bt^2}-1-at-\left(\tfrac12a^2+b\right)t^2
-\left(\tfrac16a^3+ab\right)t^3.
$$

It has nonnegative coefficients. Stable numerical evaluation uses $p=at$,
$q=bt^2$ and $P_k(z)=\sum_{j=0}^kz^j/j!$:

$$
T_4(a,b;t)
=e^p-P_3(p)+q(e^p-P_1(p))+e^p(e^q-P_1(q)).
$$

Let $Q_R$ be a sum of such tails. For any reference $T>0$ put
$C_T=Q_R(T)/T^4$. Nonnegative coefficients give
$Q_R(t)\le C_Tt^4$ for $0\le t\le T$.
When $h>0$ and $C_T>0$, set

$$
t_*=\min\left\{T,\frac{(1-\rho)h}{2C_T}\right\},
\qquad c=\frac{1+\rho}{2}<1.
$$

For every $0<t\le t_*$,

$$
\boxed{\Delta_n(t)\le c h t^3<h t^3\le\mathcal B(t).}
$$

Thus tolerance $c h t^3$ is response-feasible and is rejected by the HS
certificate. This radius is sufficient; no maximal-radius claim is made.
The formulas are analytic inequalities. Floating-point evaluations of their
parameters are numerical, without interval-certified rounding.



### 3. Arbitrary planar BV paths and fixed partitions

Let $X$ be a continuous planar BV path of Euclidean variation $L$, with fixed
partition $\pi$. On interval $i$ write its homogeneous log-signature as
$\ell_{i,1}+\ell_{i,2}+\ell_{i,3}+\cdots$. Compare exact signature with
the full series of the degree-two log-ODE decoder:

$$
F=S(X)=\prod_i S(X_i),
\qquad
G_\pi=\prod_i\exp_\otimes(\ell_{i,1}+\ell_{i,2}).
$$

Each pair of interval factors agrees through degree two and differs at degree
three by $\ell_{i,3}$. In the chronological product, multiplying this first
difference by any nonconstant factor raises degree. Consequently

$$
F_1=(G_\pi)_1,\quad F_2=(G_\pi)_2,\quad
\boxed{D_3=\sum_i\ell_{i,3}.}
$$

The local first-discarded-log identity already appears after (2.10) in
[Ferrucci, Perrée and Lyons](https://arxiv.org/html/2607.26281v1).
The response/HS comparison below additionally uses the sharp cubic gauge.

Set $a=\sum_i\lVert\ell_{i,1}\rVert_2$ and
$b=\sum_i\lVert\ell_{i,2}\rVert_{\mathrm{HS}}$.
BV factorial estimates majorize $F$ by $e^{Lt}$; tensor-product estimates
majorize $G_\pi$ by $e^{at+bt^2}$. Every planar area block has response gauge
half its HS norm (`technical.md` §3), so the decoder's matrix coefficients are
majorized by $e^{at+(b/2)t^2}$. Therefore valid tails for §1 are

$$
\boxed{
Q_R(t)=T_4(L,0;t)+T_4(a,b/2;t),\qquad
Q_H(t)=T_4(L,0;t)+T_4(a,b;t).
}
$$

This proves the sharp limit $\rho$ and explicit finite-scale separation whenever
$\sum_i\ell_{i,3}\ne0$. For coefficient radius $r$ and path dilation $s$,
substitute joint scale $t=sr$. $X$ need not be a pure block.
The result concerns log-ODE decoding and changes no universal Euler storage
asymptotics in part C.

#### Uniformity under BV perturbations

For anchored continuous BV paths $X,Y$ with lengths at most $L$ and
$\eta=\lVert X-Y\rVert_{1\text{-var}}$, multilinearity of iterated integrals
gives

$$
\lVert S_j(X)-S_j(Y)\rVert_{\mathrm{HS}}
\le\frac{\eta L^{j-1}}{(j-1)!}.
$$

One direct proof differentiates $S_j(X+u(Y-X))$ in $u\in[0,1]$. At each
insertion time, sum over the $j$ positions of $\mathrm d(Y-X)$ gives
$(L_{\mathrm{before}}+L_{\mathrm{after}})^{j-1}/(j-1)!$ against its variation
measure. Interpolated length is at most $L$. Integrating in $u$ gives the bound.

Use the degree-three logarithm identity

$$
\ell_3=S_3-\tfrac12(S_1\otimes S_2+S_2\otimes S_1)
+\tfrac13 S_1^{\otimes3}.
$$

The three displayed groups have difference bounds respectively
$\eta L^2/2$, $3\eta L^2/2$, and $\eta L^2$. Hence

$$
\lVert\ell_3(X)-\ell_3(Y)\rVert_{\mathrm{HS}}\le3\eta L^2.
$$

On a common partition, summing local bounds gives the same estimate for
$D_3=\sum_i\ell_{i,3}$: local difference variations sum to $\eta$, and
local lengths are bounded by $L$.
Also $a\le L$ and
$b\le\sum_i L_i^2\le L^2$, because
$\ell_{i,2}=S_{i,2}-S_{i,1}^{\otimes2}/2$.
Thus on any class with total length at most $L$ and $\lVert D_3\rVert\ge h_0>0$,
§2 gives a common radius using
$Q_R^*(t)=T_4(L,0;t)+T_4(L,L^2/2;t)$ and $h=h_0$.
These constants contain no partition-size term; nondegeneracy is still required.

For one baseline path with length $L_X$ and $h_X>0$, put
$L=L_X+1$ and
$\eta_* =\min\{1,h_X/(6L^2)\}$.
Every anchored perturbation within $\eta_*$ has length at most $L$ and
cubic norm at least $h_X/2$. Applying §2 with that lower norm and $Q_R^*$
gives a uniform positive scale radius. This proves perturbation stability for
ordinary BV paths, on any fixed common partition.



### 4. Two mixed displacement/area log blocks

Let $Z=[e_1,e_2]$, $v_i\in\mathbb R^2$ and
$U_i=\alpha_i Z$. Fine and coarse tensor series are

$$
F(t)=e_\otimes^{tv_1+t^2U_1}e_\otimes^{tv_2+t^2U_2},
\qquad
G(t)=e_\otimes^{t(v_1+v_2)+t^2(U_1+U_2+\frac12[v_1,v_2])}.
$$

BCH through degree three, or direct multiplication of degree-three tensor
polynomials, gives $D_3=[w,Z]$, where $\omega=\det(v_1,v_2)$ and

$$
\boxed{w=\tfrac12(\alpha_2v_1-\alpha_1v_2)
+\tfrac{\omega}{12}(v_1-v_2).}
$$

Its HS norm is $h=\sqrt6\lVert w\rVert_2$, because
$[e_1,Z]$ and $[e_2,Z]$ are orthogonal with squared norm six.
Define

$$
a_f=\lVert v_1\rVert+\lVert v_2\rVert,\quad
b_f=\sqrt2(|\alpha_1|+|\alpha_2|),\quad
a_c=\lVert v_1+v_2\rVert,\quad
b_c=\sqrt2\left|\alpha_1+\alpha_2+\tfrac12\omega\right|.
$$

Valid remainders are

$$
Q_R=T_4(a_f,b_f/2;t)+T_4(a_c,b_c/2;t),\qquad
Q_H=T_4(a_f,b_f;t)+T_4(a_c,b_c;t).
$$

Sections 1--2 prove the sharp limit and finite-scale separation for every
$w\ne0$, every $n\ge2$.

#### Perturbation stability and degeneracy

Suppose $\lVert v_i\rVert,\lVert v_i'\rVert\le V$,
$|\alpha_i|,|\alpha_i'|\le H$, and each vector/scalar parameter changes by
at most $\eta$. Bilinearity and the determinant bound give

$$
\lVert w'-w\rVert\le(H+V+V^2/2)\eta.
$$

Indeed the area/vector terms contribute $(H+V)\eta$;
$|\omega'-\omega|\le2V\eta$ and the determinant/vector terms contribute
$V^2\eta/2$. If $\lVert w\rVert\ge2\kappa$ and
$\eta\le\kappa/(H+V+V^2/2)$, then $\lVert w'\rVert\ge\kappa$.
Uniform remainder budgets are
$a_f,a_c\le2V$, $b_f\le2\sqrt2H$ and
$b_c\le\sqrt2(2H+V^2/2)$. These give a common radius by §2.

For linearly independent increments, expressing $w$ in their basis gives

$$
w=\left(\tfrac12\alpha_2+\tfrac1{12}\omega\right)v_1
-\left(\tfrac12\alpha_1+\tfrac1{12}\omega\right)v_2.
$$

Thus the cubic term vanishes exactly when
$\alpha_1=\alpha_2=-\omega/6$. This codimension-two set requires a higher-order
analysis. Pure planar area blocks have $v_1=v_2=0$ and commute exactly, so the
merge error is zero. The earlier noncommuting pure-area example used three
driver channels and falls outside this planar setting.
