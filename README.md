# early-chirp-posterior

How early in an inspiral can a neural posterior estimator, trained once on simulations, give a calibrated
posterior for a chirping signal, and how far is it from the exact posterior when only a fraction of the signal
has been observed?

The study uses a toy chirp seen through two channels with coloured Gaussian noise of known spectrum. The
likelihood is exact, so the true posterior can be sampled and used as the reference. The package is `chirpvi`
(PyTorch and numpy, nothing else).

Status: the code, the tests and the CPU smoke runs are in place. The result tables are added when the full runs
have been done on a GPU, so this README holds no result numbers yet.

## What is in the repo

```
src/chirpvi/
  config.py       constants of the problem (grid, priors, noise spectrum)
  noise.py        spectrum, Toeplitz covariance, causal whitener
  simulator.py    prior, two channel chirp, whitened observations
  smc.py          tempered sequential Monte Carlo with ensemble moves
  reference.py    exact likelihood, reference posterior, grid posterior of two angles
  splines.py      rational-quadratic splines, with a circle variant for angles
  flow.py         conditional coupling flow on the unit cube
  training.py     online training, checkpoints, posterior wrapper
  metrics.py      SBC, coverage, TARP, C2ST, mode recovery, widths
  sequential.py   reuse of the posterior at one fraction as proposal for the next
  experiments.py  the experiments behind the scripts
scripts/          calibrate_snr, check_reference, train, evaluate, sequential, report
tests/            unit tests (CPU, offline)
results/          small result files cited by the README
```

## Reproduce

Tests (CPU, under 3 minutes) and the smoke runs (tiny sizes, CPU, under 2 minutes each, written to
`results/smoke/`, which git ignores):

```
pip install -e ".[dev]"
pytest -q
python scripts/calibrate_snr.py --smoke
python scripts/check_reference.py --smoke
python scripts/train.py --smoke
python scripts/evaluate.py --smoke
python scripts/sequential.py --smoke
python scripts/report.py --smoke
```

Full runs, planned for one T4 (each writes JSON into `results/`, `report.py` prints the tables). `train.py` takes
a time budget in minutes and picks the number of steps from the speed of its first 30 steps:

```
python scripts/calibrate_snr.py --full
python scripts/check_reference.py --full
python scripts/train.py --full --minutes 25 --batch 2048
python scripts/evaluate.py --full --part calibration
python scripts/evaluate.py --full --part reference
python scripts/sequential.py --full
python scripts/evaluate.py --full --part timing
python scripts/report.py
```

## Method notes

Signal. With $\tau = t_c - t$ and $T_c = G M_c / c^3$, the Newtonian-order inspiral has phase
$\Phi(t) = \phi_c - 2 (\tau / 5 T_c)^{5/8}$, frequency $f = \frac{1}{\pi} (5/256)^{3/8} T_c^{-5/8} \tau^{-3/8}$
and amplitude $A \propto M_c^{5/4} \tau^{-1/4}$ (distance fixed). The polarisations are
$h_+ = A (1 + \cos^2 \iota) \cos\Phi / 2$ and $h_\times = A \cos\iota \sin\Phi$. Two channels see them
through a pattern that turns slowly, $\psi(t) = \alpha + \Omega t$: channel 1 is
$\cos 2\psi \, h_+ + \sin 2\psi \, h_\times$ and channel 2 is $-\sin 2\psi \, h_+ + \cos 2\psi \, h_\times$.
The five parameters are $(M_c, t_c, \phi_c, \cos\iota, \alpha)$ with uniform priors. The pair
$(\phi_c, \alpha) \to (\phi_c + \pi, \alpha + \pi/2)$ leaves both channels unchanged, so every posterior has
two equal images. A test checks this to $10^{-9}$.

Noise. Stationary Gaussian noise with a one-sided spectrum $S(f)$ that is steep below 20 Hz, flat near 35 Hz and
rising above. The autocovariance is the cosine transform of $S$, the covariance $K$ is its Toeplitz matrix,
$K = L L^T$, the noise is $L z$ and the whitener is $W = L^{-1}$. $W$ is lower triangular, so whitening the first
$n$ samples equals the first $n$ entries of the whitened full stream. This makes the log likelihood exact for
every observed fraction, and one evaluation gives all of them through a cumulative sum.

Observation at fraction $f$. The first $f N$ samples of a buffer of $N = 256$ samples at 128 Hz (2 s); the merger
lies 0.2 to 0.5 s after the end of the full buffer.

Reference sampler. Tempered sequential Monte Carlo along $\pi_\beta \propto \text{prior} \times L^\beta$ with
adaptive steps (effective sample size kept at a set share of the particles), systematic resampling, and affine
invariant stretch moves (Goodman and Weare); angles use the shortest wrapped difference. After every stage the
exact pair symmetry is applied to a random half of the particles, which is a valid move and gives the two images
equal weight; `check_reference.py` also runs the sampler without it and reports how far it lands. Checks in the
tests: a bimodal Gaussian mixture with known weights and evidence, a periodic bump across the boundary,
importance weights from a wrong proposal, the symmetry move on a two-mode target, and the exact grid posterior of
$(\phi_c, \alpha)$.

Flow. A coupling flow on the unit cube with a uniform base, rational-quadratic splines, and a residual MLP
embedding of the whitened strain and $f$. Angles are circular: the spline has equal slopes at both ends of the
circle and learned shifts around it, and the conditioners see $(\cos, \sin)$. Training uses fresh simulations at
every step with the number of observed samples uniform between 32 and 256.

Sequential variant. The posterior at one fraction is the proposal for the next. The likelihood of the new samples
gives importance weights (exact, because the whitener is causal), and the same sampler tempers and moves from
there. The start can be the sampler's own posterior or the flow, whose density corrects the first weights.

Metrics. SBC ranks of seven fixed functions of the parameters and of the exact log likelihood, HPD and central
interval coverage, TARP, a classifier two-sample test against the reference (two folds, held out accuracy), mode
weights for the pair symmetry, and posterior widths against $f$. Definitions are in the docstrings of
`metrics.py`.

## Limits

- Toy Newtonian waveform: no spins, no merger and ringdown, no higher harmonics, distance fixed.
- No real detector response: two ideal channels, a known rotating pattern, one sky angle.
- Coloured Gaussian noise with a spectrum that the likelihood knows exactly; real analyses estimate it and meet
  glitches. The exact likelihood makes the reference exact, which a real search does not have.
- $f$ is a fraction of a fixed 2 s buffer, not of the unknown total time to merger.
- One architecture and a small number of training runs; no test outside the training distribution.
- The reference sampler is checked on analytic and grid problems and by agreement between independent runs, not
  proved correct on every observation; ensemble moves may mix slowly on thin curved ridges, and its evidence
  estimate is noisier than its samples.

## License

Apache-2.0.
