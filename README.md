# early-chirp-posterior

How early in an inspiral can a neural posterior estimator, trained once on simulations, give a calibrated
posterior for a chirping signal, and how far is it from the exact posterior when only a fraction of the signal
has been observed?

The study uses a toy chirp with a known coloured noise spectrum, so the exact posterior of the Gaussian
likelihood can be sampled and used as the reference. The package is `chirpvi`.

## What is in the repo

```
src/chirpvi/   simulator, whitening, reference sampler, spline flow, training, metrics
tests/         unit tests (CPU, offline)
scripts/       smoke and full runs that write JSON into results/
results/       small result files cited by the README
```
