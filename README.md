# PEX-CI

Reference implementation of **PEX-CI**, an identity-exact net-load forecaster with a covariate-conditioned,
channel-independent patch Transformer, from the manuscript *Identity-Exact Net-Load Forecasting with a
Covariate-Conditioned Patch Transformer* (M. Massaoudi and M. Ez Eddin).

The net load is `NL = L - PV - WD` (load minus photovoltaic and wind generation). PEX-CI forecasts the three
components and defines the net-load forecast through this identity, so the balance holds exactly and no penalty
weight is needed.

## Model

| Element | Implementation (`pex_ci.py`) |
|---|---|
| Identity-exact output | `PEXOutput`: `NL_hat = L_hat - PV_hat - WD_hat` |
| Bounded PV head | `PV_hat = C * sigmoid(r) * TOA/1361`, non-negative, zero at night, `C = softplus(c~) * sigma_NL` |
| Instance normalisation | window mean/std per channel (std floor 1e-2), statistics passed to the decoder |
| Encoder | `CIEnc`: shared-weight channel-independent patch Transformer (patch 16, stride 8, post-LN layers) |
| Decoder | `CovDecoder`: direct head + linear covariate path + FiLM-type interaction with the covariates of every horizon step |
| Skip | persistence/AR skip per component; PV skip in clear-sky-index space; optional operator forecast as base level (`op_residual=True`) |
| Loss | `pex_loss`: `MSE(NL_hat) + beta * MSE(components)`; `pex_loss_soft` is the soft-penalty control |
| Intervals | `conformal_quantiles`, `conformal_interval`: split-conformal quantiles of validation residuals per horizon step |

All power series share one scale: `x~ = (x - mu_c) / sigma_NL`, with `mu_c` the training mean of channel `c` and
`sigma_NL` the training standard deviation of `NL`. Hence the identity also holds in normalised units.

### Frozen hyper-parameters (`FINAL_HPARAMS`, `FINAL_TRAIN`)
Selected on validation data only (Optuna TPE, 20 trials, 12 epochs).

| | |
|---|---|
| Encoder | width 64, 3 layers, 8 heads, token reduction 16, dropout 0.3 (history weather channels included) |
| Decoder | FiLM, hidden width 128, dropout 0.3, analog covariates on the linear path only |
| Optimiser | AdamW, lr 5.23e-4, weight decay 6.06e-6, cosine schedule, batch 64, at most 20 epochs |
| Early stopping | patience 6 on validation net-load MSE, gradient clipping 5 |
| Loss weight | beta = 0.523 |

With these settings the model has between 0.7 M parameters (systems without wind, `T = 3`) and 1.1 M parameters
(with wind, `T = 4`) for a 96-step horizon; the count depends on the horizon and on the number of covariate columns.
Systems without a wind series use `T = 3` and the output channels `[NL, L, PV]`.

## Tensor layout

`forward(x, F, Fh)` with all power quantities in the shared normalised space:

* `x`  `[B, L, C]` history window. Channels: `[NL, L, PV, WD, humidity, TOA irradiance, surface irradiance, air density, hour sin, hour cos, weekday sin, weekday cos]`, i.e. `T = 4` power channels first, then 4 weather and 4 time channels.
* `F`  `[B, H, Cf]` target-aligned covariates, all known at the forecast origin:
  columns 0-7 are hour sin/cos, weekday sin/cos, TOA irradiance (z-score), TOA/1361, weekly load profile (out-of-fold on training rows) and holiday flag;
  then `2*nc` analog columns (component values at the same time on the most recent observed day, then week);
  then, optionally, `nc` operator day-ahead forecasts and one staleness flag.
* `Fh` `[B, L, 8]` the base covariates at the history steps.
* output `[B, H, 4]`: `[NL_hat, L_hat, PV_hat, WD_hat]`.

`PEXSpec` collects the few data-dependent constants (column indices, scales, shapes) the model needs.

## Usage

```python
import torch
from pex_ci import PEXSpec, build_pex_ci, pex_loss

spec = PEXSpec(T=4, L=96, H=96, C=12, Cf=8 + 6, Cf_base=8, toa_unit_idx=5, anch_idx=6,
               pv_mu=120.0, sig=800.0, pv_max_n=3.0, analog_idx=8)
model = build_pex_ci(spec)                      # frozen configuration; pass e.g. use_fk=False to switch covariates off

x, F, Fh = torch.randn(16, 96, 12), torch.randn(16, 96, 14), torch.randn(16, 96, 8)
F[..., 5] = F[..., 5].abs() * 0.5               # TOA/1361 in [0, 1]
Fh[..., 5] = Fh[..., 5].abs() * 0.5
y = model(x, F, Fh)                             # [16, 96, 4]
assert torch.allclose(y[..., 0], y[..., 1] - y[..., 2] - y[..., 3], atol=1e-5)   # exact identity

loss = pex_loss()(y, torch.randn_like(y))
loss.backward()
```

Training follows the settings above (`FINAL_TRAIN`). Net-load intervals are obtained by calling `conformal_quantiles`
on the validation forecasts and `conformal_interval` on the test forecasts.

## Requirements
Python 3.9+ and PyTorch 2.x. NumPy is needed only for the conformal helpers.

## License
MIT (see `LICENSE`).
