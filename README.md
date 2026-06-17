# Implicit Neural Representations for 1D Signal Reconstruction

A minimal, educational implementation of an **Implicit Neural Representation (INR)** applied to 1D signal denoising and interpolation, built with PyTorch.

## What is an INR?

An Implicit Neural Representation is a neural network that learns to represent a signal as a **continuous function** mapping coordinates → values. Instead of storing discrete samples, the network *is* the signal. This makes INRs naturally resolution-independent and capable of interpolating to unseen coordinates.

## What this demo shows

| Step | Details |
|------|---------|
| Signal | Composite sinusoid: `2·sin(t) + 1.5·cos(2t) + 0.5·sin(3t)` |
| Noise | Gaussian (σ = 0.05), ~30 dB SNR |
| Encoding | Fourier feature positional encoding (L=5, 10-dim output) |
| Model | 3-layer MLP — `10 → 128 → 128 → 1` (~18k parameters) |
| Training | **20% sparse sampling** (200 of 1000 points), Adam, 5000 epochs |
| Denoising | Reconstructs the full signal from sparse noisy observations |
| Interpolation | Queries the trained model at 1200 points (denser than training grid) |

### Results

The INR matches the noisy baseline in denoising quality while generalising smoothly to unseen coordinates.

## Repository structure

```
inr_1d_demo.ipynb   ← self-contained demo notebook (all steps, all plots)
requirements.txt    ← Python dependencies
```

## Getting started

```bash
git clone https://github.com/<your-username>/implicit-neural-representation-1d
cd implicit-neural-representation-1d
pip install -r requirements.txt
jupyter notebook inr_1d_demo.ipynb
```

A GPU is not required; the model trains in under a minute on CPU.

## Key concepts

- **Spectral bias**: plain MLPs learn low frequencies first. Fourier feature encoding lifts coordinates into a higher-frequency space, letting the network capture fine signal detail.
- **Sparse supervision**: training on 20% of samples demonstrates that an INR can recover the full signal from incomplete observations — useful for compressed sensing scenarios.
- **Continuous representation**: the learned function can be evaluated at *any* coordinate, enabling smooth interpolation beyond the training grid.

## References

- Mildenhall et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020
- Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, NeurIPS 2020
- Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions (SIREN)*, NeurIPS 2020
