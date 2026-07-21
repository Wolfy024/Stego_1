# Result validation report

**Decision:** Share with caveats

**Scope:** HiNet-warm-started, 100-step gates-only checkpoint; 200-pair benchmark

**Data discovered:** 5,100 images across COCO val2017 and DIV2K validation

## Checks

| Check | Result | Evidence |
|---|---:|---|
| Benchmark/comparison reconciliation | PASS | Exact within 1e-9 dB |
| Dataset audit | PASS | Readable, unique, disjoint split |
| Checkpoint tensor shapes | PASS | 256×256 cover and secret |
| Independent PSNR calculation | PASS | max Δ=0.00e+00 dB |
| Aggregate cover target | PASS | >=38 dB |
| Aggregate secret target | PASS | >=35 dB |
| Novel gates updated | PASS | changed parameters=120,461 |
| JPEG limitation exposed | PASS | all real-JPEG PSNR values <12 dB |
| Numerical stability | PASS | non-finite steps=0 |

## Reconciled headline metrics

| Metric | Value |
|---|---:|
| Cover → stego PSNR | 38.553 dB |
| Cover → stego SSIM | 0.9631 |
| Secret recovery PSNR (clean) | 36.347 dB |
| Secret recovery SSIM (clean) | 0.9602 |
| Secret recovery PSNR (real JPEG QF-50) | 11.227 dB |
| Secret recovery SSIM (real JPEG QF-50) | 0.0093 |

## Caveats

- Results use one seeded 90/10 split; there are no multi-seed confidence intervals.
- Only 3/200 validation images are from DIV2K; source-specific claims are unsupported.
- Real JPEG destroys recovery (about 11.2 dB); this checkpoint is clean-channel only.
- The invertible backbone uses disclosed public HiNet weights; this is not training from scratch.
- Local adaptation updates 123,024 attention/band-gate parameters for 100 steps.
- Similarity percentage is MAE-derived fidelity, not exact-pixel extraction accuracy.
- No external baseline was reimplemented, so superiority claims are out of scope.
