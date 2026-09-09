# Neural differential path experiments

Status, 8 September: experiments A and B are complete. Experiment A comprises
the fixed-path adequacy and signature audits, the paired 10,000-update loss
comparison, numerical checks and the hundred-block local-signature refinement;
results are reconstructed in `../notebooks/05_neural_ode_path.ipynb`.
Experiment B comprises the accepted Brownian-driver-to-OU-response Neural CDE
comparison in `../notebooks/06_brownian_ou_operator.ipynb`. Section 6 records an
unrun nonlinear extension rather than completed evidence. Forecasting remains
outside scope.

## 1. Aim

Hold data and model class fixed, vary discrepancy used for training, and measure
how learned output path changes. For discrepancy $D$, fitted parameters are

$$
\widehat\theta_D
=\operatorname*{arg\,min}_\theta
\frac1n\sum_{i=1}^n
D\!\left(F_\theta(X^{(i)}),Y^{(i)}\right).
$$

$X^{(i)}$ is input, $Y^{(i)}$ is target, and
$\widehat Y_D^{(i)}=F_{\widehat\theta_D}(X^{(i)})$ is fitted output. Every loss
uses same inputs and targets. Loss changes fitted parameters and may therefore
change output paths.

Two experiments form one progression.

| experiment | input | target | fitted output | role |
|---|---|---|---|---|
| A: one-path fit | time and learned initial state | one fixed path $Y^\star$ | Neural ODE path $\widehat Y$ | expose loss-dependent approximation |
| B: stream-to-stream calibration | Brownian path $W^{(i)}$ | paired OU response $Y^{(i)}$ | Neural CDE response $\widehat Y^{(i)}$ | test transfer to causal operator learning |

Experiment A is path fitting. Experiment B is path-to-path operator learning.

Metric axioms alone do not show that a discrepancy represents structure useful
for a learning task. Every discrepancy is therefore judged by fitted paths and
outcomes outside its own training objective. The $(\epsilon,r)$ or $d_r$ family
is one possible geometry, with $r$ selecting response scale; its validity as a
metric does not establish empirical relevance.

## 2. Discrepancies

Let target times satisfy

$$
0=t_0<t_1<\cdots<t_{C-1}=T,
$$

and residuals be $e_r=\widehat Y(t_r)-Y(t_r)$.

### 2.1 Pointwise MSE

$$
D_{\mathrm{MSE}}(\widehat Y,Y)
=\frac1C\sum_{r=0}^{C-1}\|e_r\|_2^2.
$$

Each recorded target has equal weight.

### 2.2 Elapsed-time $J_2$

Define trapezoidal weights

$$
w_0=\frac{t_1-t_0}{2},
\qquad
w_r=\frac{t_{r+1}-t_{r-1}}{2},
\qquad
w_{C-1}=\frac{t_{C-1}-t_{C-2}}{2},
$$

where middle expression applies for $1\leq r\leq C-2$. Since
$\sum_r w_r=T$, normalized loss is

$$
D_{J_2}(\widehat Y,Y)
=\frac1T\sum_{r=0}^{C-1}w_r\|e_r\|_2^2.
$$

Each residual is weighted by elapsed time it represents.

### 2.3 Anchored coordinate-mean signature loss

Let $Z_Y$ be output path augmented by normalized time. If output dimension is
$d_Y$, then $Z_Y:[0,T]\to\mathbb R^{d_Z}$ with $d_Z=d_Y+1$. Level $k$
signature is

$$
S^{(k)}(Z_Y)
=\int_{0<u_1<\cdots<u_k<T}
\mathrm dZ_Y(u_1)\otimes\cdots\otimes\mathrm dZ_Y(u_k).
$$

For truncation depth $L$, initial experiment uses

$$
D_{\mathrm{coord},L}(\widehat Y,Y)
=\frac1{d_Y}\|\widehat Y(0)-Y(0)\|_2^2
+\sum_{k=1}^L d_Z^{-k}
\left\|S^{(k)}(Z_{\widehat Y})-S^{(k)}(Z_Y)\right\|_2^2.
$$

Anchor is required because signature is unchanged by constant translation of
output channel. Factors $1/d_Y$ and $d_Z^{-k}$ make each block a mean squared
coordinate discrepancy: level $k$ has $d_Z^k$ coordinates. This is an
engineering convention rather than the response-derived $d_r$ family derived
in notebook 05 §5.
It does not compensate factorial decay or equalize gradients. For continuous
bounded-variation $Z$ with channelwise $\ell^1$ variation $V_Z$,

$$
\|S^{(k)}(Z)\|_1\leq\frac{V_Z^k}{k!}.
$$

Time is normalized by $T$. Experiment A provisionally uses identity output
scaling because constructed output is dimensionless and order one. Levelwise
value and parameter-gradient audit may revise this before training. Resulting
rule applies to prediction and target and is frozen before comparative
signature runs. Later operator tasks require their own training-data rule.

Response-derived alternative is

$$
D_{r,L}(\widehat Y,Y)
=\|\widehat Y(0)-Y(0)\|_2
+\sum_{k=1}^L r^k
\left\|S^{(k)}(Z_{\widehat Y})-S^{(k)}(Z_Y)\right\|_1.
$$

Its weights and tensor norm follow the upper bound for systems with response
strength at most $r$. It enters an experiment only when application supplies a
meaningful $r$. Relation $rV_p\approx L$ is a scale heuristic; a truncation rule
also needs target tail accuracy. Initial comparison uses $D_{\mathrm{coord},L}$
and makes no $d_r$ claim.

Coordinate $\ell^1$ is required by response triangle inequality. Since level
$k$ has $d_Z^k$ coordinates,

$$
\|\Delta S^{(k)}\|_1
\leq d_Z^k\|\Delta S^{(k)}\|_\infty.
$$

Replacing $\ell^1$ by $\ell^\infty$ without factor $d_Z^k$ defines a different
loss and loses stated response upper bound.

Every training signature is computed from predicted and target values at the
same target times $t_r$, with piecewise-linear interpolation between them. Time
augmentation, interpolation, scaling, grid and truncation depth are fixed
across losses. Dense-grid signatures are evaluation metrics only. This prevents
signature training from accessing target values unavailable to other losses.

### 2.4 Smooth-only Sobolev comparator

For paths $Y,\widehat Y\in H^1([0,T];\mathbb R^{d_Y})$, define

$$
D_{H^1}(\widehat Y,Y)
=D_{J_2}(\widehat Y,Y)
+\frac{\rho}{T}\sum_{r=0}^{C-1}w_r
\|\dot{\widehat Y}(t_r)-\dot Y(t_r)\|_2^2.
$$

$\rho>0$ fixes relative value and derivative scales before training. For
experiment A set

$$
\bar Y^\star=\frac1T\sum_{r=0}^{C-1}w_rY^\star(t_r),
\qquad
\rho=
\frac{\sum_r w_r\|Y^\star(t_r)-\bar Y^\star\|_2^2}
{\sum_r w_r\|\dot Y^\star(t_r)\|_2^2}.
$$

This balances target value variation and derivative energy without looking at a
fitted model. Experiment A has analytic $\dot Y^\star$ and a differentiable
Neural ODE output, so it may use this loss as a secondary local-shape
comparator. Brownian, OU and
double-well sample paths are almost surely outside $H^1$; $D_{H^1}$ is excluded
from experiments B and 6. A finite-difference derivative penalty on those paths
would depend on grid resolution and amplify stochastic increments.

## 3. Experiment A: one target path

### 3.1 Aim

Given one restricted continuous path model, measure which aspects of one fixed
target are preserved by MSE, $J_2$, global signature loss, and local signature
loss.

### 3.2 Target path

Set $T=1$. Define $Y^\star:[0,1]\to\mathbb R^2$ by

$$
Y_1^\star(t)=\cos(2\pi t)+a(t)\cos(12\pi t),
$$

$$
Y_2^\star(t)=\sin(2\pi t)+a(t)\sin(12\pi t),
$$

where

$$
a(t)=0.3\exp\!\left[-100(t-0.25)^2\right].
$$

Base terms form one large loop. Local term adds smaller loops near $t=0.25$.
Target is fixed before fitting.

Reference grid is

$$
\tau_m=\frac{m}{M},
\qquad m=0,\ldots,M,
\qquad M=512.
$$

Dense reference values $Y^\star(\tau_m)$ are used for evaluation. Training uses
$C=64$ target values.

### 3.3 Target observation conditions

Uniform condition uses

$$
t_r^{\mathrm{unif}}=\frac{r}{C-1},
\qquad r=0,\ldots,C-1.
$$

Clustered condition uses

$$
u_r=\frac{r}{C-1},
\qquad
t_r^{\mathrm{cluster}}=1-(1-u_r)^3.
$$

It places more observations near $t=1$ and fewer near local event at $t=0.25$.
Every loss uses identical target times within each condition.

### 3.4 Neural ODE models

For $K=5$, define Fourier time features

$$
\gamma(t)
=\left[t,\sin(2^k\pi t),\cos(2^k\pi t)\right]_{k=0}^{K-1}.
$$

They reduce risk that raw-time spectral bias decides whether model can represent
localized oscillation. Evidence for Fourier features concerns coordinate
networks rather than this vector field, so an equal-budget adequacy check is
required before comparison. Hidden path $h:[0,1]\to\mathbb R^H$ satisfies

$$
h(0)=\eta,
\qquad
\dot h(t)=f_\theta(\gamma(t),h(t)),
\qquad
\widehat Y(t)=g_\phi(h(t)).
$$

$\eta\in\mathbb R^H$ is learned initial hidden state. Vector field is a
two-layer tanh MLP and decoder $g_\phi:\mathbb R^H\to\mathbb R^2$ is affine.
Trainable parameters are $\Theta=(\theta,\phi,\eta)$.

Use two capacities:

| capacity $c$ | hidden dimension $H_c$ | vector-field width | role |
|---|---:|---:|---|
| restricted | 2 | 16 | expose approximation trade-offs |
| expressive | 8 | 64 | check whether loss effects vanish near exact fit |

Use fixed-step RK4 with step $1/512$. This keeps numerical path and evaluation
grid aligned. Capacity varies as an experimental factor and remains fixed across
losses within each comparison.

### 3.5 Paired initialization

Use seeds $s\in\{0,1,2,3,4\}$. For each capacity $c$ and seed $s$, initialize
one parameter vector $\Theta_{0}^{(c,s)}$ and clone it for every loss:

$$
\Theta_{D,0}^{(c,s)}=\Theta_0^{(c,s)}.
$$

Observation times, solver, optimizer and update budget are also shared.

### 3.6 Training algorithm

Use full-batch Adam with learning rate $10^{-3}$ for 10,000 updates. This is a
fixed compute horizon rather than a convergence claim. Record common metrics at
updates $0,100,500,1000,2500,5000,7500,10000$. For capacity
$c$, loss $D$, sampling condition $q$, and seed $s$:

1. Load target pairs
   $\{(t_r^q,Y^\star(t_r^q))\}_{r=0}^{C-1}$.
2. Build capacity $c$ and initialize
   $\Theta\leftarrow\Theta_0^{(c,s)}$.
3. For update $k=0,\ldots,9999$:
   1. solve $\dot h=f_\theta(\gamma(t),h)$ from $0$ to $1$;
   2. evaluate $h(t_r^q)$ at all target times;
   3. decode $\widehat Y(t_r^q)=g_\phi(h(t_r^q))$;
   4. calculate $D(\widehat Y,Y^\star)$;
   5. differentiate through decoder and ODE solver;
   6. update $\Theta$ with Adam.
4. Save final parameters, loss history, dense predictions and configuration.

Initial signature pilot uses seed zero and uniform observations at both
capacities. It compares MSE, $J_2$, global signature and local signature losses.
Uniform sampling prevents representation geometry from being confounded with
observation density. Extension across seeds follows only if both signature
losses optimize with finite gradients and informative checkpoint trajectories.

Before those runs, verify under an unused pilot seed that expressive model can
fit dense target under MSE and that Fourier time features improve over raw time
under same update budget. Freeze feature map after this check. Run $D_{H^1}$
only as a secondary uniform-observation comparator, with $\rho$ fixed from
training-target scales before optimization. It does not gate signature runs.

### 3.7 Evaluation

Solve each fitted model at every $\tau_m$. Report:

$$
E_{\mathrm{MSE}}
=\frac1{M+1}\sum_{m=0}^{M}
\|\widehat Y(\tau_m)-Y^\star(\tau_m)\|_2^2,
$$

$$
E_{J_2}
=\frac1T\sum_{m=0}^{M}\widetilde w_m
\|\widehat Y(\tau_m)-Y^\star(\tau_m)\|_2^2,
$$

$$
E_{H^1}
=E_{J_2}
+\frac{\rho}{T}\sum_{m=0}^{M}\widetilde w_m
\|\dot{\widehat Y}(\tau_m)-\dot Y^\star(\tau_m)\|_2^2,
$$

$$
E_\infty
=\max_{0\leq m\leq M}
\|\widehat Y(\tau_m)-Y^\star(\tau_m)\|_2,
$$

and local-event error

$$
E_{\mathrm{local}}
=\frac1{0.2}\int_{0.15}^{0.35}
\|\widehat Y(t)-Y^\star(t)\|_2^2\,\mathrm dt.
$$

Evaluate every model under every discrepancy. Plot target, observations, each
fitted path, both coordinates against time, and residual magnitude against
time.

### 3.8 Global against local signatures

For $Z_Y=(t,Y_1,Y_2)$, $d_Z=3$. Global depth four contains

$$
3+3^2+3^3+3^4=120
$$

coordinates. Split $[0,1]$ into ten equal intervals and calculate depth two on
each interval. Local representation contains

$$
10(3+3^2)=120
$$

coordinates. Compare global depth four with local depth two under equal feature
count. Equal count controls stored representation size; it does not equalize
compute, gradient scale or information content. Interpret result as a
depth-versus-localisation comparison under matched storage.

Precisely, global loss is $D_{\mathrm{global}}=D_{\mathrm{coord},4}$. For
partition $I_j=[(j-1)/10,j/10]$, local loss is

$$
D_{\mathrm{local},2,10}(\widehat Y,Y)
=\frac1{d_Y}\|\widehat Y(0)-Y(0)\|_2^2
+\frac1{10}\sum_{j=1}^{10}\sum_{k=1}^{2}3^{-k}
\left\|
S^{(k)}(Z_{\widehat Y}|_{I_j})-S^{(k)}(Z_Y|_{I_j})
\right\|_2^2.
$$

Factor $1/10$ averages interval contributions. Local blocks attach features to
coarse time regions and omit direct cross-interval words. Global time-augmented
signature retains cross-interval interactions and encodes timing through mixed
time-output words.

#### 3.8.1 Finer local partition closeout

The ten-block comparison confounds two choices: how local the representation is
and the numerical scale of its signature levels. To isolate localisation, the
closeout refines the partition to $K=100$ blocks while retaining $K_0=10$ as a
scale reference. It trains with

$$
\begin{aligned}
D_{\mathrm{local},2,K}^{(K_0)}(\widehat Y,Y)
={}&\frac1{d_Y}\|\widehat Y(0)-Y(0)\|_2^2\\
&+\frac1K\sum_{j=1}^{K}\sum_{k=1}^{2}
3^{-k}\left(\frac{K}{K_0}\right)^{2k}
\left\|
S^{(k)}(Z_{\widehat Y}|_{I_j})
-S^{(k)}(Z_Y|_{I_j})
\right\|_2^2,
\qquad K=100,\quad K_0=10.
\end{aligned}
$$

Multiplier is a project normalization derived from standard homogeneity of
iterated integrals (Lyons, Caruana and Lévy, 2007). If an augmented local
increment is rescaled by $c$, then $S^{(k)}(cZ)=c^kS^{(k)}(Z)$. Shortening a
smooth block from length $1/K_0$ to
$1/K$ therefore shrinks a level-$k$ discrepancy approximately by $K_0/K$ to
the power $k$, and its squared norm by the power $2k$. Multiplication by
$(K/K_0)^{2k}$ removes this automatic shrinkage. On straight paths the
correction makes the levelwise $K=100$ and $K=10$ losses exactly equal; curved
paths can still differ because the finer partition retains different local
information.

Training still observes the same 64 values and treats their piecewise-linear
interpolant as the path. Each of the 101 partition boundaries is evaluated by
piecewise-linear interpolation. This adds no observations: it compares the same
observed polygon after finer localization. Since 100 blocks exceed
the 63 observed line segments, many blocks lie inside a single segment. The
refinement may therefore behave mainly like localized increment matching.
Interpretation must account for this increasingly increment-like behaviour.
Writing $e=\widehat Y-Y$, level-one difference on block $I_j$ is
$e(j/K)-e((j-1)/K)$. For differentiable $e$,

$$
\frac1K\sum_{j=1}^{K}\frac13
\left(\frac{K}{K_0}\right)^2
\left\|e(j/K)-e((j-1)/K)\right\|_2^2
\longrightarrow
\frac{1}{3K_0^2}\int_0^1\|\dot e(t)\|_2^2\,\mathrm dt.
$$

Fine level one is therefore a finite-difference derivative penalty, coupled
with anchor and level-two terms. Existing $H^1$ runs provide a necessary
control for this interpretation.

Only six new fits are required: three paired seeds for each of the restricted
and expressive models, all trained for 10,000 Adam updates at learning rate
$10^{-3}$. MSE, $J_2$, $H^1$, global signature and ten-block local signature
fits are reused as fixed controls. Every stored path is reevaluated under both
the original ten-block loss and the scaled hundred-block loss, in addition to
dense MSE, $J_2$, $H^1$, maximum error and local-event $J_2$.

The initial audit reports both the raw hundred-block loss and the scaled
hundred-block loss. The raw value demonstrates the shrinkage caused by shorter
blocks; only the scaled version is trained. Improvement only on the scaled
hundred-block objective is objective recovery. Repeated improvement in the
independent local-event or pointwise metrics supports a useful localization
effect. Failure to improve, or severe optimization instability, means the
ten-block result is not repaired simply by using more intervals.

### 3.9 Outcome meanings fixed before run

| outcome | interpretation |
|---|---|
| $J_2$ improves sparse region only under clustered observations | elapsed-time weighting mechanism supported |
| MSE and $J_2$ nearly tie under uniform observations | expected control; trapezoidal endpoint half-weights prevent exact equality |
| all fits approach target exactly | model capacity removes loss-dependent approximation |
| loss effects appear at restricted capacity and vanish at expressive capacity | capacity-dependent loss selection supported |
| both capacities show same loss effect | discrepancy changes optimization or inductive bias beyond simple capacity limit |
| $H^1$ improves derivative error only | derivative supervision recovers its local objective without broader path benefit |
| $H^1$ improves local-event or signature metrics | local derivative control explains part of apparent signature benefit |
| signature fit wins only on signature metric | objective recovery only |
| signature fit improves loop or local-event recovery under independent metrics | path representation has broader effect |
| all losses fail | model, solver or optimization must be repaired before comparison |

## 4. Experiment B: Brownian stream to OU stream

### 4.1 Aim

Verify data generation, causality, gradients and learned simultaneous stream
transformation $W\mapsto Y$ on a known linear system. MSE and $J_2$ supply an
operator calibration. OU is not the main signature evidence.

### 4.2 Dataset

Use $T=1$, $M=256$, $\Delta t=T/M$, and fine times
$\tau_m=m\Delta t$. Split sizes are

$$
n_{\mathrm{train}}=512,
\qquad
n_{\mathrm{val}}=128,
\qquad
n_{\mathrm{test}}=256.
$$

For each path $i$, draw independent

$$
\xi_m^{(i)}\sim\mathcal N(0,1),
\qquad m=0,\ldots,M-1,
$$

and define

$$
\Delta W_m^{(i)}=\sqrt{\Delta t}\,\xi_m^{(i)},
\qquad
W_0^{(i)}=0,
\qquad
W_{m+1}^{(i)}=W_m^{(i)}+\Delta W_m^{(i)}.
$$

Fix $\lambda=2$, $\sigma=0.5$, and $y_0=0$. Paired OU target is

$$
Y_0^{(i)}=y_0,
$$

$$
Y_{m+1}^{(i)}
=Y_m^{(i)}-\lambda Y_m^{(i)}\Delta t
+\sigma\Delta W_m^{(i)}.
$$

One example is $(W^{(i)},Y^{(i)})$. Input and target cover same interval.

### 4.3 Input and target observations

Model initially receives complete driver
$\{(\tau_m,W_m^{(i)})\}_{m=0}^{M}$. Input missingness and irregularity are
deferred, so output loss is only varying mechanism.

Use $C=64$ target times. Uniform and clustered definitions match experiment A.
Piecewise-linear interpolation of generated $Y^{(i)}$ supplies target values at
these times. Every paired loss run uses same drivers, OU targets and target
times.

### 4.4 Neural CDE model

Define control

$$
X^{(i)}(t)=(t,W^{(i)}(t))\in\mathbb R^2
$$

using piecewise-linear interpolation. Hidden dimension is $H=16$. Shared
initial hidden state is $h^{(i)}(0)=\eta\in\mathbb R^{16}$. Hidden path solves

$$
\mathrm dh^{(i)}(t)
=V_{\theta,0}(h^{(i)}(t))\,\mathrm dt
+V_{\theta,1}(h^{(i)}(t))\,\mathrm dW^{(i)}(t).
$$

Vector field $V_\theta:\mathbb R^{16}\to\mathbb R^{16\times2}$ is a two-layer
tanh MLP with width 64. Decoder $q_\phi:\mathbb R^{16}\to\mathbb R$ is affine:

$$
\widehat Y^{(i)}(t)=q_\phi(h^{(i)}(t)).
$$

Use fixed-step RK4 with one solver step per control interval. This model differs
from current `LinearCDEQuery`: it decodes hidden trajectory at each time rather
than decoding all times from final summary.

This experiment learns a deterministic operator on piecewise-linear controls at
the fixed grid. Under grid refinement, ordinary differential equations driven
by piecewise-linear Brownian approximations have a Stratonovich limit. Constant
diffusion makes Itô and Stratonovich forms agree for stated OU and double-well
targets. Learned $V_{\theta,1}(h)$ may be state-dependent, so its parameters are
not interpreted as an Itô diffusion coefficient. Recovering SDE coefficients is
outside this experiment.

### 4.5 Training algorithm

Use seeds $s\in\{0,1,2\}$, batch size 64, Adam with learning rate $10^{-3}$,
and 500 epochs. Each epoch has eight training batches, hence 4,000 updates.
Batch order is fixed per seed and shared across losses.

For loss $D$, target condition $q$, and seed $s$:

1. Generate train, validation and test pairs once from split-specific seeds.
2. Generate target times once; share them across loss runs.
3. Initialize one model and clone parameters for each loss.
4. For each minibatch $\mathcal B$:
   1. interpolate controls $X^{(i)}=(t,W^{(i)})$ for $i\in\mathcal B$;
   2. solve Neural CDE along each control;
   3. decode $\widehat Y^{(i)}(t_r)$ at target times;
   4. calculate

      $$
      L_D
      =\frac1{|\mathcal B|}\sum_{i\in\mathcal B}
      D(\widehat Y^{(i)},Y^{(i)});
      $$

   5. backpropagate through decoder and CDE solver;
   6. update $\theta$, $\phi$, and $\eta$ with Adam.
5. Evaluate validation metrics after each epoch without changing model.
6. Save fixed-budget final model. Test split remains unused until design and
   configurations are fixed.

Initial operator comparison uses MSE and $J_2$ under uniform and clustered
targets. Signature losses follow after MSE learning, causality and gradients are
verified. Signature training follows §2.3: predictions and targets are sampled
at the same $C$ target times and joined by piecewise-linear interpolation.

### 4.6 Acceptance checks

Required before loss comparison:

1. generated $Y^{(i)}$ satisfies stated Euler recurrence exactly;
2. model output shape is $(B,C,1)$ and gradient reaches every parameter;
3. changing driver changes predicted response;
4. changing driver after time $\tau$ leaves predictions before $\tau$ unchanged
   to solver tolerance;
5. one small batch can be fitted closely under MSE;
6. held-out error falls below untrained error;
7. identical seeds reproduce paths, target times, initialization and batch
   order across losses.

### 4.7 Test evaluation

For each test pair, decode prediction at all fine-grid times. Primary metric is

$$
E_{\mathrm{fine},J_2}
=\frac1{n_{\mathrm{test}}T}
\sum_{i=1}^{n_{\mathrm{test}}}
\sum_{m=0}^{M}\widetilde w_m
|\widehat Y_m^{(i)}-Y_m^{(i)}|^2.
$$

Also report fine-grid MSE, $J_1$, $J_4$, $L^\infty$, signature discrepancies,
runtime and peak memory.

Independent OU dynamics residual is

$$
R_m^{(i)}
=\widehat Y_{m+1}^{(i)}-\widehat Y_m^{(i)}
+\lambda\widehat Y_m^{(i)}\Delta t
-\sigma\Delta W_m^{(i)},
$$

$$
E_{\mathrm{dyn}}
=\frac1{n_{\mathrm{test}}M}
\sum_{i=1}^{n_{\mathrm{test}}}
\sum_{m=0}^{M-1}|R_m^{(i)}|^2.
$$

For target condition $q$ and seed $s$, paired loss effect is

$$
\delta_{q,s}
=E_{q,s}^{J_2\text{-trained}}
-E_{q,s}^{\mathrm{MSE\text{-trained}}}.
$$

Sampling-mechanism contrast is

$$
\Delta_s=\delta_{\mathrm{clustered},s}-\delta_{\mathrm{uniform},s}.
$$

Elapsed-time weighting mechanism predicts $\Delta_s<0$.

### 4.8 OU signature diagnostic

OU output is one-dimensional, so augment it as $Z_Y=(t,Y)\in\mathbb R^2$.
Global depth four has

$$
2+2^2+2^3+2^4=30
$$

coordinates. Depth two on five equal intervals also has

$$
5(2+2^2)=30
$$

coordinates. An optional diagnostic may compare MSE, $J_2$, global depth four,
and local depth two after MSE and $J_2$ operator runs pass acceptance checks.
Matched coordinate count controls storage only, as in §3.8.

OU is linear in driver and lies inside a small controlled model class, so near
exact agreement under all losses is expected. Time augmentation still carries
order: an early Brownian increment is damped longer than a late increment, and
mixed time-driver or time-output signature terms distinguish them. OU therefore
checks signature implementation and optimization, while nonlinear system in §6
supplies main operator comparison.

### 4.9 Outcome meanings fixed before run

| outcome | interpretation |
|---|---|
| $J_2$ advantage grows under clustered targets | sampling-weight result transfers to operator learning |
| MSE and $J_2$ tie with near-zero error | OU is exact-learnability check; stronger task required |
| $J_2$ wins equally under both target conditions | mechanism may be optimization rather than sampling measure |
| all signature rows tie near zero | expected OU acceptance result; no conclusion about signature usefulness |
| signature training wins only on signature discrepancy | diagnostic objective recovery only |
| causality or one-batch fit fails | implementation fault; comparative run is invalid |

## 5. Signature implementation checks

For piecewise-linear increment $\Delta Z$, segment signature is

$$
\left(1,\Delta Z,\frac{\Delta Z^{\otimes2}}{2!},\ldots,
\frac{\Delta Z^{\otimes L}}{L!}\right).
$$

Concatenate segments using Chen identity

$$
S_{\mathrm{new}}^{(k)}
=\sum_{j=0}^k S_{\mathrm{old}}^{(j)}\otimes
\frac{\Delta Z^{\otimes(k-j)}}{(k-j)!}.
$$

Differentiable PyTorch implementation requires tests against:

1. straight-line closed form;
2. Chen concatenation computed by two routes;
3. `iisignature` on fixed paths;
4. zero loss on identical paths;
5. nonzero gradient under perturbation;
6. translation invariance before anchor and sensitivity after anchor;
7. levelwise loss and parameter-gradient magnitudes recorded on a fixed
   perturbed batch before training. Scaling is frozen before paired runs.

## 6. Nonlinear operator comparison (future design; unrun)

OU operator is linear and lies close to model class. Retain it as implementation
check and use nonlinear target dynamics

$$
\mathrm dY_t
=\alpha(Y_t-Y_t^3)\,\mathrm dt+\sigma\,\mathrm dW_t.
$$

Choose $(\alpha,\sigma,T)$ from the fixed grid

$$
\alpha\in\{0.5,1,2,4\},
\qquad
\sigma\in\{0.5,0.8,1.2,1.6\},
\qquad
T\in\{1,2\},
$$

thirty-two combinations, evaluated on generated paths only. Wells sit at
$Y=\pm1$ with barrier height $\alpha/4$, so crossing frequency is governed by
$\exp(-\alpha/2\sigma^2)$ and $\sigma$ must be comparable with the barrier for
crossings to occur at all. Select a regime where
20% to 80% of targets cross zero, and among qualifying combinations take the one
with crossing fraction closest to 0.5. Record the full grid table. Freeze
parameters before model training. Initial state is $Y_0 = -1$, placing every
path in the left well so that crossing is a property of the driver rather than
of initialisation. Generate paired targets by

$$
Y_{m+1}^{(i)}
=Y_m^{(i)}
+\alpha\left(Y_m^{(i)}-(Y_m^{(i)})^3\right)\Delta t
+\sigma\Delta W_m^{(i)}.
$$

Reuse Neural CDE, splits, target conditions and paired training procedure of §4.
Train under MSE, $J_2$, global coordinate-mean signature loss and local
coordinate-mean signature loss.

The sFML benchmark uses $\alpha=1$, $\sigma=0.5$ and learns a one-step
state-transition law which is iterated over long horizons. This experiment
learns the paired same-interval map $W\mapsto Y$. Matching its parameters would
not make results directly comparable. Parameter selection here instead ensures
enough crossings for first-crossing and occupation metrics. A low-noise,
long-horizon metastable regime is a separate extension.

**Capacity sweep.** Argument of §3.4 applies here unchanged: a separating
discrepancy reaches the same solution once the model represents the operator
exactly, so a null result at one capacity cannot distinguish an irrelevant loss
from an over-large model. Use two hidden dimensions with vector-field widths
scaled alongside:

| capacity $c$ | hidden $H_c$ | vector-field width | role |
|---|---:|---:|---|
| restricted | 4 | 32 | expose approximation trade-offs |
| expressive | 16 | 64 | check whether loss effects vanish near exact fit |

Loss is compared within each fixed capacity, and the pair of capacities is the
measurement: an effect present at 4 and absent at 16 identifies
capacity-dependent approximation, while a null at both is evidence only after
both models pass §6.1.

### 6.1 Generator acceptance checks

Required before any §6 training run, matching §4.6 for the linear case:

1. generated $Y^{(i)}$ satisfies the stated cubic Euler recurrence exactly;
2. every generated path is finite, and $\max_{i,m}|Y^{(i)}_m|$ stays below 5.
   Explicit Euler on $\alpha(Y - Y^3)$ is unstable when $|\alpha(1-3Y^2)\Delta t|$
   approaches 2, so a grid point that lets $|Y|$ grow is rejected rather than
   integrated more finely;
3. crossing fraction on the training split lies in $[0.2, 0.8]$ as selected;
4. checks 2 to 7 of §4.6 hold with the cubic generator substituted.

Define first upward crossing

$$
\tau(Y)=\inf\{t\in[0,T]:Y_t\geq0\},
$$

with $\tau(Y)=T$ when no crossing occurs. Define positive-well occupation

$$
A(Y)=\frac1T\int_0^T\mathbf 1_{\{Y_t>0\}}\,\mathrm dt.
$$

Independent test outcomes are

$$
E_\tau
=\frac1{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
|\tau(\widehat Y^{(i)})-\tau(Y^{(i)})|,
$$

and

$$
E_A
=\frac1{n_{\mathrm{test}}}
\sum_{i=1}^{n_{\mathrm{test}}}
|A(\widehat Y^{(i)})-A(Y^{(i)})|.
$$

Signature training improving only its own discrepancy is objective recovery.
Improvement in $E_\tau$ or $E_A$ supports preservation of path behaviour not
directly optimized.

## 7. Execution record and evidence locations

The completed sequence is:

1. **Experiment A, fixed-path comparison.** Fourier adequacy, signature
   value-gradient audits, paired 10,000-update fits, three-seed closeout,
   learning-rate sensitivity, quadrature and solver-resolution checks, and the
   hundred-block local-signature refinement are complete. The analysis in
   `../notebooks/05_neural_ode_path.ipynb` reads the stored configurations,
   histories, fitted models and metrics from
   `../results/runs/neural_ode_fixed_path_*`.
2. **Experiment B, causal operator calibration.** Generator and model
   acceptance, paired MSE and $J_2$ comparisons under uniform and clustered
   target sampling, and uniform-grid signature comparisons are complete for
   three seeds. The executable analysis and exact evidence map are in
   `../notebooks/06_brownian_ou_operator.ipynb`; stored artefacts are under
   `../results/runs/neural_cde_brownian_ou/`.
3. **Brownian message sensitivity.** This is a separate path-space distance
   experiment, not the nonlinear operator design of §6. Its procedure and
   completed analysis are in
   `../notebooks/07_brownian_message_sensitivity.ipynb`.

The nonlinear double-well operator in §6, the deferred classification branch
and signature-Wasserstein comparisons are optional future extensions. They are
not needed for the report's completed evidence chain.

## 8. Historical design review

The following review was written on 22 August, before implementation. It is
retained because it explains why controls and acceptance gates were added;
§7 and notebooks 05–07 give the current execution status and results.
Present and future tense below belongs to that dated design discussion. Actual
execution differed in one material respect: experiment B retained its signature
calibration rows, while nonlinear §6 was left unrun.

Procedure covers the meeting notes closely: experiment A is the neural-ODE path
fit with learned vector field and learned initial hidden state, experiment B is
the Brownian-to-OU stream task, forecasting and signature-Wasserstein are
excluded, and §3.8 and §4.8 pose the depth-against-intervals question at
matched stored coordinate count (120 against 120, 30 against 30). This controls
representation size, while compute and information remain unmatched. Section 1
records distinction between metric validity and task-relevant structure.
Best fit under different Banach norms remains deferred exposition.

### Confirmed

- **Target design.** Base loop plus a burst at $t = 0.25$, in two dimensions, so
  Lévy area is non-trivial and the signature has order content to see.
- **Clustered condition starves the local event.** $t_r = 1 - (1-u_r)^3$ maps
  $[0.15, 0.35]$ to roughly 8% of $u$, so about 5 observations against 13 under
  uniform. $E_{\mathrm{local}}$ then measures where the mechanism predicts an
  effect, which completes the chain from mechanism to metric.
- **Paired initialisation** (§3.5), shared times, solver, optimiser and budget.
- **Outcome meanings fixed before running** (§3.9, §4.9), which 20/08 lacked.
- **Causality check** (§4.6 item 4): changing the driver after $\tau$ must leave
  predictions before $\tau$ unchanged. Sharper than anything proposed on 20/08.
- **$E_{\mathrm{dyn}}$** (§4.7) is an independent outcome optimised by no loss
  under comparison.
- **Nonlinear follow-up** (§6) with parameters frozen before model training,
  and independent outcomes in first crossing time and occupation time.

### Signature weighting

§2.3 uses

$$\sum_{k=1}^{L} d_Z^{-k}\big\|\Delta S^{(k)}\big\|_2^2 ,$$

so weights are $3^{-k}$ in A and $2^{-k}$ in B, with squared level norms.
Notebook 05 §5 instead derives unsquared $\lambda_k=r^k$ weights in tensor
$\ell^1$ from a response-strength bound $r$.

Section 2.3 therefore labels $D_{\mathrm{coord},L}$ as a coordinate-average
engineering loss and keeps response-derived $D_{r,L}$ separate. Initial-value
anchor is averaged over its $d_Y$ coordinates. Levelwise magnitude and gradient
audit in §5 checks numerical balance before training.

### Experiment B has a calibration role

OU is affine in $Y$, so its linear-CDE form uses constant channel
$\overline Y=(Y,1)^\top$. Time and driver matrices do not commute; mixed
time-driver words record when an increment arrived. OU remains weak evidence
for signature loss because its driver-to-response map is linear and exactly
representable by a small controlled model.

Combined with a 16-dimensional Neural CDE containing a one-dimensional linear
truth, experiment B is expected to be null under every loss. §4.9 anticipates
this for MSE against $J_2$; the same holds a priori for the signature rows, and
saying so prevents a foregone null being read as evidence.

Section 7 therefore keeps OU as MSE acceptance and uses nonlinear system of §6
for main operator loss comparison.

### Capacity is fixed where it should vary

20/08 establishes that separating discrepancies share the same zero-loss
solution once model represents operator exactly. A null result at one capacity
cannot distinguish excessive capacity from an irrelevant loss. Sections 3.4 and
6 therefore use restricted and expressive settings.

### Spectral bias risk in experiment A

$f_\theta(\gamma(t),h)$ must produce a $12\pi$ oscillation localised near
$t=0.25$. Fourier map reduces raw-time spectral bias, while §3.6 requires an
equal-budget check because evidence for this device comes from coordinate MLPs.

### Resolved changes

1. Signature coordinate weighting is an engineering convention; response-based
   $r^k$ weighting is separate.
2. Nonlinear system carries operator comparison; OU is acceptance check.
3. Experiments A and 6 use capacity sweeps.
4. Vector field receives Fourier time features subject to adequacy check.
5. OU signature rows are optional implementation diagnostics.

## 9. Response to review

Response recorded 22 August before implementation. Operative procedure in §§1
to 7 incorporates accepted changes.

### Signature weighting: accepted with qualification

Previous $d_Z^{-k}\|\Delta S^{(k)}\|_2^2$ was an unstated coordinate-average
convention and had no $d_r$ derivation. Section 2.3 now names it
$D_{\mathrm{coord},L}$, divides anchor by its coordinate count, and states that
it is an engineering loss. Response-derived alternative

$$
D_{r,L}
=\|\Delta Y_0\|_2
+\sum_{k=1}^L r^k\|\Delta S^{(k)}\|_1
$$

is recorded separately. It is used only when application supplies response
bound $r$. Relation $rV_p\approx L$ gives scale, while rigorous truncation also
requires target tail tolerance; it does not uniquely determine $r$.

### OU signature role

OU is linear in $W$ and exactly representable by a small controlled model, so it
is weak evidence for choosing a signature loss. Procedure now uses OU for data,
causality, gradient and MSE acceptance, and moves main signature comparison to
nonlinear system.

With augmented state $\overline Y=(Y,1)^\top$,

$$
\mathrm d\overline Y_t
=A_0\overline Y_t\,\mathrm dt+A_1\overline Y_t\,\mathrm dW_t,
$$

where

$$
A_0=
\begin{pmatrix}-\lambda&0\\0&0\end{pmatrix},
\qquad
A_1=
\begin{pmatrix}0&\sigma\\0&0\end{pmatrix}.
$$

These matrices do not commute:

$$
A_0A_1=
\begin{pmatrix}0&-\lambda\sigma\\0&0\end{pmatrix},
\qquad
A_1A_0=0.
$$

An early driver increment is damped longer than a late increment. Mixed
time-driver signature terms therefore record relevant order even though map is
linear in $W$. Signature loss also compares complete time-augmented output
paths, rather than only endpoint $Y_T$. Expected null arises from simplicity and
exact learnability, not absence of all order information.

### Capacity sweep: accepted

Experiment A now uses restricted $(H=2,\text{ width}=16)$ and expressive
$(H=8,\text{ width}=64)$ Neural ODEs. Loss is compared within each fixed
capacity. Effect appearing in restricted model and vanishing in expressive
model identifies capacity-dependent approximation. A null at both capacities
remains evidence only after both models pass optimization checks.

### Spectral bias: accepted

Vector field now receives shared Fourier map $\gamma(t)$ with five dyadic
frequencies. This reduces risk that raw scalar time makes local $12\pi$
component unavailable to every loss. Equal-budget adequacy check in §3.6 tests
whether features help this vector field.

### Metric structure: accepted

Section 1 now states that metric validity does not imply task-relevant geometry.
Every loss is evaluated through fitted paths and independent outcomes.

### Affine best fit: deferred

Best affine approximation under different Banach norms is a useful analytic
illustration of capacity-dependent projection. It remains outside immediate
execution order because it neither fits neural differential path nor learns a
stream-to-stream operator. It may enter report exposition without delaying
experiments.

## 10. Changes made when closing the review gaps

Initial second-reader edits on 22 August established §6 capacity and parameter
controls.

**Capacity sweep added to §6.** Argument accepted for experiment A in §9 applies
to the nonlinear operator comparison unchanged, and §6 previously reused one
fixed capacity. Restricted $(H=4,\text{ width}=32)$ and expressive
$(H=16,\text{ width}=64)$, compared within each capacity. Without the pair, a
null in §6 cannot separate an irrelevant loss from an over-large model, and §6
is where the main comparison now sits.

**Parameter grid written down.** §6 previously named a selection rule with no
grid. Grid is now $\alpha\in\{0.5,1,2,4\}$, $\sigma\in\{0.5,0.8,1.2,1.6\}$,
$T\in\{1,2\}$, with the qualifying combination closest to crossing fraction 0.5
selected and the full table recorded. Initial state fixed at $Y_0=-1$ so that
crossing is a property of the driver.

**Grid corrected after simulation.** First grid proposed here was
$\alpha\in\{2,5,10\}$, $\sigma\in\{0.3,0.5,0.8\}$. Simulated on 256 paths at
$\Delta t = 1/256$ it produced a maximum crossing fraction of 0.24, so the
20% to 80% window was barely reachable and most combinations gave no crossings
at all. Wells lie at $Y=\pm1$ with barrier $\alpha/4$, and Kramers rate scales
as $\exp(-\alpha/2\sigma^2)$, so large $\alpha$ with small $\sigma$ suppresses
crossing entirely. Replacement grid lowers $\alpha$ and raises $\sigma$;
simulation gives fourteen qualifying combinations, with $(\alpha,\sigma,T) =
(4, 1.2, 2)$ closest to 0.5 at crossing fraction 0.55.

**§6.1 generator acceptance checks added.** §4.6 covers the linear generator and
had no cubic counterpart. New checks: exact cubic recurrence, finiteness with
$\max|Y| < 5$, crossing fraction in the selected window, and §4.6 items 2 to 7
with the cubic generator substituted. Explicit Euler on $\alpha(Y-Y^3)$ is
unstable once $|\alpha(1-3Y^2)\Delta t|$ approaches 2; across the replacement
grid the observed maximum is 3.26, so the bound is a guard rather than a
constraint, and a grid point violating it is rejected rather than integrated
more finely.

## 11. Literature check, 08/20 logs onward and this procedure

Independent primary-source search completed 22 August. Search found adjacent
work for every component, but no exact matched comparison of MSE, elapsed-time
$J_2$, global signature and local signature losses for paired Neural ODE or
Neural CDE path outputs. This is evidence that comparison is underexplored,
rather than proof of novelty.

### Sobolev training

[Sobolev training for neural networks](https://arxiv.org/abs/1706.04859) and
[Sobolev training for operator learning](https://arxiv.org/abs/2402.09084)
combine value and derivative information. [NDO-NODE](https://arxiv.org/abs/2106.04166)
is closer to trajectory fitting: it supervises state and derivative information
when learning ODE dynamics.

$H^1$ compares derivatives at matched times, so it is locally sensitive to
direction and timing. It lacks higher-order cross-time interactions represented
by signatures. Its cost also includes target derivatives and differentiation
through predicted derivatives.

Brownian and diffusion sample paths are almost surely outside $H^1$; see
[Brownian path regularity](https://arxiv.org/abs/2202.10114). Section 2.4
therefore limits $H^1$ to smooth experiment A. It is a secondary comparator,
not a requirement that signature loss must outperform.

### Neural ODE loss literature

[Neural ODEs](https://arxiv.org/abs/1806.07366) establish continuous-depth
parameterisation. Searches found derivative supervision and weak residual
training, but no exact matched study using this procedure's four losses.

[Weak Penalty Neural ODE](https://arxiv.org/abs/2511.06609) integrates a
dynamical-equation residual against test functions. Its objective has form

$$
\int_0^T
\big(\dot{\widehat Y}(t)-f_\theta(\widehat Y(t),t)\big)\phi(t)\,\mathrm dt,
$$

whereas $J_2$ integrates prediction error
$\|\widehat Y(t)-Y(t)\|_2^2$. Weak residual training is adjacent work on
identifying dynamics from noisy data; it is not equivalent to elapsed-time
weighting of output error.

[Fourier features](https://arxiv.org/abs/2006.10739) improve high-frequency
learning in coordinate MLPs. They motivate $\gamma(t)$ but do not prove benefit
inside a Neural ODE vector field. Section 3.6 therefore requires an
equal-budget adequacy check.

### Distributional and individual-path signature comparisons

Distributional signature objectives include
[conditional Sig-Wasserstein](https://arxiv.org/abs/2006.05421) and
[signature-moment MMDs](https://jmlr.org/beta/papers/v23/20-1466.html). This is
the dominant generative-model line.

Individual-path signature geometry also exists.
[Local regression on path spaces with signature metrics](https://arxiv.org/abs/2510.16728)
uses signature semimetrics between individual paths for regression and
classification. [Signatures Meet Dynamic Programming](https://arxiv.org/abs/2312.05547)
uses deviation from reference signatures in trajectory following and control.
These works do not provide the controlled supervised neural path-output
comparison in §§3 and 6, but they rule out a claim that pathwise signature
comparison itself is new.

The supported claim is narrower: direct paired signature loss for supervised
path-output training, compared under matched models and seeds with MSE and
$J_2$, appears underexplored in searched literature.

### Path construction and signature scaling

A signature belongs to an interpolated path. [Path Imputation Strategies for
Signature Models](https://arxiv.org/abs/2005.12359) shows that path construction
can materially change model performance. [Embedding and Learning with
Signatures](https://arxiv.org/abs/1911.13211) likewise finds that augmentation
and embedding choices matter.

Section 2.3 now fixes target grid, piecewise-linear interpolation, time
augmentation, scaling and depth. Training signatures use the same target values
as pointwise losses. Coordinate weights $d_Z^{-k}$ remain an engineering
normalisation. Section 5 records levelwise loss and gradient magnitudes before
freezing that choice.

Finite truncation plus an initial-value anchor remains a task representation,
and may identify distinct paths. Metric validity and task relevance are kept
separate.

### Global and local signatures

[Concise $(\varepsilon,r)$ representations](https://arxiv.org/abs/2607.26281)
studies a trade-off between signature depth and interval subdivision under a
fixed storage budget. It supports the comparison in §§3.8 and 4.8.

Full-interval signatures need not lose useful local information:
[Embedding and Learning with Signatures](https://arxiv.org/abs/1911.13211)
reports successful full-domain representations in its studied feature tasks.
Global against local performance is therefore empirical. Equal coordinate
count controls storage, while compute, scaling and information remain unequal.

### Neural CDE and Brownian-driver semantics

[Neural CDEs](https://arxiv.org/abs/2005.08926) support using a controlled hidden
path for irregular stream input and pathwise output decoding. The architecture
matches the operator aim $W\mapsto Y$.

[Wong-Zakai convergence](https://arxiv.org/abs/0808.0337) gives a Stratonovich
limit for differential equations driven by smooth Brownian approximations.
Piecewise-linear controls in §4 therefore require an interpretation. Constant
diffusion makes Itô and Stratonovich targets agree in the OU and double-well
examples. General learned driver fields need not share that property. Section
4.4 defines a fixed-grid piecewise-linear path operator and avoids interpreting
learned fields as Itô SDE coefficients.

### Double-well task

[Learning Stochastic Dynamical System via Flow Map
Operator](https://arxiv.org/abs/2305.03874) uses

$$
\mathrm dX_t=(X_t-X_t^3)\,\mathrm dt+0.5\,\mathrm dW_t
$$

to learn a one-step stochastic transition law from state trajectories and
iterate it over long horizons. Section 6 instead learns a paired same-interval
driver-to-response map. Matching $(\alpha,\sigma,T)$ would not produce a direct
benchmark comparison.

Current parameter selection seeks enough crossings for first-crossing and
occupation metrics. Higher noise and shorter horizon are appropriate for that
purpose. Low-noise long-horizon metastability is a distinct extension.

### Classification

Signature classification predates this project.
[Generalised Signature Method](https://arxiv.org/abs/2006.00873) benchmarks
signature feature pipelines across multivariate time-series datasets.
[Local regression on path spaces with signature metrics](https://arxiv.org/abs/2510.16728)
also compares signature semimetrics with $L^p$, supremum and dynamic-time-warping
distances.

Planned classification pipeline remains useful as a controlled test of fixed
1-nearest-neighbour distance, interpolation and missingness. Its contribution
is controlled comparison under project data conditions, rather than first use
of signatures for classification.

### Procedure consequences

| finding | operative change |
|---|---|
| Sobolev training is established but requires smooth paths | $H^1$ added only to experiment A as secondary comparator |
| weak residual and $J_2$ optimize different quantities | weak-form work treated as adjacent rather than equivalent |
| exact four-loss comparison was not found | novelty language limited to underexplored comparison |
| individual-path signature metrics already exist | claim narrowed to supervised paired path-output training |
| signatures depend on constructed path | training grid, interpolation, augmentation and scaling fixed in §2.3 |
| depth and interval subdivision trade storage against representation | matched feature count described as matched storage only |
| piecewise-linear Brownian controls have Stratonovich limit | §4.4 states fixed-grid operator interpretation |
| sFML learns a different operator | §6 parameters selected for crossings, not superficial comparability |
| signature classification has prior art | classification retained as controlled evaluation, not novelty claim |

The completed progression is experiment A followed by the OU operator
calibration in experiment B. The nonlinear stream-to-stream design and
classification study remain possible future work, while Brownian message
sensitivity is reported separately in notebook 07. Literature changed protocol
details and the strength of the claims without changing the central question:
how a path discrepancy affects learning or comparison of path-valued objects.
