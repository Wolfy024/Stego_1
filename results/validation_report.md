# Result validation report

**Decision:** Share with caveats

**Scope:** Wavelet-gated INN, 100 local adapter steps; 200-pair aggregate benchmark

**Data discovered:** 5,100 images across the declared benchmark roots

## Checks

| Check | Result | Evidence |
|---|---:|---|
| Selected-model reconciliation | PASS | Wavelet-gated INN; 6 named fields exact within 1e-9 |
| Dataset audit | PASS | Readable, unique, disjoint split |
| Checkpoint tensor shapes | PASS | 256×256 cover and secret |
| Checkpoint FP32 forward | PASS | stego=torch.float32, revealed=torch.float32 |
| Independent PSNR calculation | PASS | max Δ=0.00e+00 dB |
| Corrected clean FP32 target | PASS | precision='float32 tensors (autocast disabled; legacy TF32 setting unrecorded)'; cover=38.604, secret=36.955 dB |
| Local adapters updated | PASS | changed shared adapter parameters=120,461 |
| JPEG limitation exposed | PASS | all five real-JPEG PSNR values are present and <12 dB |
| Numerical stability | PASS | non-finite steps=0 |
| Aggregate candidate-vs-upstream Pareto | PASS | cover=+0.0201 dB, secret=+0.3730 dB |
| Paired evidence integrity | PASS | n=200, mean=+0.3898 dB, 95% CI=[+0.2910, +0.4981] |
| Paired evaluation protocol | PASS | precision='float32 (autocast disabled, TF32 disabled)' |
| Paired candidate-vs-upstream Pareto | PASS | cover=+0.0191 dB, secret=+0.3898 dB |
| Paired checkpoint identity | PASS | baseline=8dd45ddef55e…, candidate=3fe383ecbfda… |

## Reconciled headline metrics

| Metric | Value |
|---|---:|
| Evaluation precision | float32 tensors (autocast disabled; legacy TF32 setting unrecorded) |
| Cover → stego PSNR | 38.604 dB |
| Cover → stego SSIM | 0.9635 |
| Secret recovery PSNR (clean) | 36.955 dB |
| Secret recovery SSIM (clean) | 0.9658 |
| Secret recovery PSNR (real JPEG QF-50) | 11.227 dB |
| Secret recovery SSIM (real JPEG QF-50) | 0.0093 |
| Aggregate cover PSNR delta vs upstream | +0.0201 dB |
| Aggregate secret PSNR delta vs upstream | +0.3730 dB |
| Paired cover PSNR mean delta vs upstream | +0.0191 dB |
| Paired secret PSNR mean delta vs upstream | +0.3898 dB |

## Caveats

- The paired bootstrap covers one fixed 200-pair cohort; it does not measure training-seed variability.
- Only 3/200 benchmark images are from DIV2K_valid_HR; source-specific claims are unsupported.
- Real JPEG destroys recovery (about 11.2 dB
  at QF-50); this checkpoint is clean-channel only.
- The invertible backbone uses disclosed public HiNet weights; this is not training from scratch.
- Local adaptation updates 123,024 declared adapter parameters.
- Similarity percentage is MAE-derived fidelity, not exact-pixel extraction
  accuracy.
- No classical steganography baseline was reimplemented on the paired cohort; broad
  superiority claims are out of scope.
