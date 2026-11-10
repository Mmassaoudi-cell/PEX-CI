"""PEX-CI: identity-exact net-load forecasting with a covariate-conditioned, channel-independent patch Transformer.

Net load is defined through the balance NL = L - PV - WD, so the identity holds exactly by construction (no penalty weight).
PV is produced as a bounded fraction of the astronomical (top-of-atmosphere) envelope. A decoder conditions every horizon
step on target-aligned covariates (calendar, sun geometry, holiday flag, weekly profile, same-time analogs and, when
published, operator day-ahead forecasts). All power series share one scale (training standard deviation of NL).

Tensor conventions (all power quantities in the shared normalised space)
    x   [B, L, C]    history window; channels = [NL, L, PV, WD, 4 weather, 4 time]   (T = 4 power targets first)
    F   [B, H, Cf]   target-aligned covariates (see the README for the column layout)
    Fh  [B, L, Cf0]  base covariates at the history steps (Cf0 = columns shared with F[..., :Cf0])
    out [B, H, T]    channels = [NL_hat, L_hat, PV_hat, WD_hat]  with  NL_hat = L_hat - PV_hat - WD_hat  exactly

The model needs a data object `spec` with the attributes listed in `PEXSpec` (the dataclass below is a ready-made container).
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn


# ------------------------------------------------------------------------------------------------ data description
@dataclass
class PEXSpec:
    """Everything the model needs to know about the data layout (normalised units, see README)."""
    T: int                      # power channels incl. net load: 4 = [NL, L, PV, WD]
    L: int                      # lookback (steps)
    H: int                      # horizon (steps)
    C: int                      # input channels of x: T + 4 weather + 4 time
    Cf: int                     # columns of the target-aligned covariate tensor F
    Cf_base: int                # leading base columns of F (calendar, TOA, weekly profile, holiday)
    toa_unit_idx: int           # column of TOA/1361 in F (physical envelope of PV)
    anch_idx: int               # column of the weekly load profile in F
    pv_mu: float                # training mean of PV (MW)
    sig: float                  # shared power scale: training standard deviation of NL (MW)
    pv_max_n: float             # largest training PV value in normalised units
    nc: Optional[int] = None    # number of components (T - 1)
    analog_idx: Optional[int] = None   # first analog column in F (None: no analogs)
    op_idx: Optional[int] = None       # first operator-forecast column in F (None: no operator forecasts)

    def __post_init__(self):
        if self.nc is None: self.nc = self.T - 1


# ------------------------------------------------------------------------------------------------ building blocks
class EncLayer(nn.Module):
    """Post-LN Transformer encoder layer: Z' = LN(Z + MHA(Z)),  Z+ = LN(Z' + FFN(Z'))."""
    def __init__(self, D, heads, ff, drop):
        super().__init__()
        self.att = nn.MultiheadAttention(D, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(D, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, D))
        self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D); self.d = nn.Dropout(drop)

    def forward(self, x):
        a, _ = self.att(x, x, x, need_weights=False); x = self.n1(x + self.d(a))
        return self.n2(x + self.d(self.ff(x)))


def patchify(x, P, S, N):
    """x [B,L,C] -> [B,N,C,P] with tail padding; N = floor((L-P)/S) + 2."""
    x = x.transpose(1, 2)
    x = torch.cat([x, x[..., -1:].expand(-1, -1, S)], -1).unfold(-1, P, S)[:, :, :N]      # [B,C,N,P]
    return x.permute(0, 2, 1, 3)


class CIEnc(nn.Module):
    """Channel-independent shared-weight patch Transformer (PatchTST style) over the component (and optional weather)
    channels of the instance-normalised window; the time channels are excluded (they are future-known covariates)."""
    def __init__(self, C, L, d=128, P=16, S=8, layers=3, heads=8, drop=0.2, ds=16):
        super().__init__()
        self.idx = list(range(1, C - 4)); self.Cs = len(self.idx)          # components (+ weather); no NL (derived), no time
        self.P, self.S, self.N = P, S, (L - P) // S + 2
        self.emb = nn.Linear(P, d)
        self.cemb = nn.Parameter(torch.randn(self.Cs, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(self.N, d) * 0.02)
        self.layers = nn.ModuleList([EncLayer(d, heads, 2 * d, drop) for _ in range(layers)])
        self.red = nn.Linear(d, ds); self.dim = self.Cs * self.N * ds; self.drop = nn.Dropout(drop)

    def forward(self, x):
        B = x.shape[0]
        p = patchify(x[..., self.idx], self.P, self.S, self.N).permute(0, 2, 1, 3)     # [B,Cs,N,P]
        z = self.drop(self.emb(p) + self.pos + self.cemb).reshape(B * self.Cs, self.N, -1)
        for l in self.layers: z = l(z)
        z = z.reshape(B, self.Cs, self.N, -1)
        return self.red(z).flatten(1)


class CovDecoder(nn.Module):
    """Direct linear head (latent -> H) + linear covariate path + FiLM-type interaction between a latent-conditioned code
    and the covariates of every horizon step. mode 'linear' keeps only the two linear paths."""
    def __init__(self, din, H, nc, Cf, dh=64, drop=0.1, mode="film", film_cols=None):
        super().__init__()
        self.nc, self.dh, self.mode, self.H = nc, dh, mode, H
        self.film_cols = film_cols                       # None: FiLM sees all covariates; k: only the first k columns
        self.direct = nn.Linear(din, H * nc)
        self.flin = nn.Linear(Cf, nc)
        nn.init.zeros_(self.flin.weight); nn.init.zeros_(self.flin.bias)
        if mode == "film":
            self.u = nn.Linear(din, dh * nc)
            self.q = nn.Sequential(nn.Linear(film_cols or Cf, dh), nn.GELU(), nn.Linear(dh, dh))
            self.drop = nn.Dropout(drop)
            self.W1 = nn.Parameter(torch.randn(nc, dh, dh) / math.sqrt(dh)); self.b1 = nn.Parameter(torch.zeros(nc, dh))
            self.W2 = nn.Parameter(torch.randn(nc, dh) / math.sqrt(dh)); self.b2 = nn.Parameter(torch.zeros(nc))

    def forward(self, z, F):                              # z [B,din]  F [B,H,Cf]
        B = z.shape[0]
        d = self.direct(z).reshape(B, self.H, self.nc)
        if self.mode == "linear": return d + self.flin(F)
        d = d + self.flin(F)
        u = self.u(z).reshape(B, 1, self.nc, self.dh)
        q = self.q(F if self.film_cols is None else F[..., : self.film_cols]).unsqueeze(2)         # [B,H,1,dh]
        h = self.drop(Fn.gelu(u + q))
        h = Fn.gelu(torch.einsum("bhcd,cde->bhce", h, self.W1) + self.b1)
        g = torch.einsum("bhce,ce->bhc", h, self.W2) + self.b2                                      # [B,H,nc]
        return d + g


class PEXOutput(nn.Module):
    """Maps the component pre-activations to the 4-channel (NL, L, PV, WD) forecast with the exact identity."""
    def __init__(self, spec, use_anchor=False, pv_physics=True, exact=True):
        super().__init__()
        self.use_anchor, self.pv_physics, self.exact = use_anchor, pv_physics, exact
        self.toa_i, self.anch_i = spec.toa_unit_idx, spec.anch_idx
        self.mu_pv, self.sig = spec.pv_mu, spec.sig
        cap0 = (spec.pv_mu + spec.sig * spec.pv_max_n) * 1.1                 # MW, 10 % above the largest training observation
        self.cap = nn.Parameter(torch.tensor(math.log(math.expm1(cap0 / spec.sig))))      # softplus(.) * sig = capacity C (MW)

    def forward(self, r, F):                              # r [B,H,nc], F [B,H,Cf]
        L = r[..., 0] + (F[..., self.anch_i] if self.use_anchor else 0.0)
        if self.pv_physics:
            cap = Fn.softplus(self.cap) * self.sig
            pv_mw = cap * torch.sigmoid(r[..., 1]) * F[..., self.toa_i]       # bounded by the astronomical envelope, 0 at night
            PV = (pv_mw - self.mu_pv) / self.sig
        else:
            PV = r[..., 1]
        comps = [L, PV] + [r[..., k] for k in range(2, r.shape[-1])]
        NL = L - PV - sum(comps[2:]) if len(comps) > 2 else L - PV            # exact identity
        return torch.stack([NL] + comps, -1)


def prep_cov(F, Fh, x, spec, m=None, sd=None):
    """Express the weekly profile relative to its value at the origin and the analog / operator columns relative to the
    window (mean and standard deviation of the matching power channel)."""
    Fin = F.clone()
    Fin[..., spec.anch_idx] = F[..., spec.anch_idx] - Fh[:, -1:, spec.anch_idx]
    if m is None:
        m = x[..., : spec.nc + 1].mean(1); sd = x[..., : spec.nc + 1].std(1) + 1e-2
    else:
        m, sd = m[:, 0], sd[:, 0]
    for c in range(spec.nc):
        ch = c + 1; mc, sc = m[:, ch].unsqueeze(1), sd[:, ch].unsqueeze(1)
        cols = []
        if spec.analog_idx is not None: cols += [spec.analog_idx + c, spec.analog_idx + spec.nc + c]
        if spec.op_idx is not None: cols += [spec.op_idx + c]
        for col in cols: Fin[..., col] = (F[..., col] - mc) / sc
    return Fin


# ------------------------------------------------------------------------------------------------ the model
class PEX(nn.Module):
    """PEX-CI forecaster.  r_c = encoder/decoder correction + persistence/AR skip.
       L, WD : skip = window mean + linear map of the demeaned history (level persistence + linear AR)
       PV    : skip in clear-sky-index space (kappa = PV / (C0 * TOA/1361)), acting as a logit offset of the PV head
       encoder sees the instance-normalised history (RevIN style) and the window statistics enter the decoder."""
    def __init__(self, spec, enc_kw=None, dec_dh=64, drop=0.2, use_anchor=False, pv_physics=True, exact=True,
                 use_fk=True, lin_skip=True, inst_norm=True, rel_anchor=True, fk_mode="film", use_wx=False, fk_mask=(),
                 film_base_only=False, op_residual=False):
        super().__init__()
        self.fk_mask = list(fk_mask)                      # zero these covariate columns in the decoder input (ablation)
        self.T, self.L, self.H, self.C = spec.T, spec.L, spec.H, spec.C
        self.keep = list(range(spec.T)) + (list(range(spec.T, spec.T + 4)) if use_wx else []) + list(range(spec.C - 4, spec.C))
        self.Ce = len(self.keep)
        self.nc = spec.T - 1
        self.use_fk, self.exact, self.lin_skip, self.inst_norm, self.rel_anchor = use_fk, exact, lin_skip, inst_norm, rel_anchor
        self.enc = CIEnc(self.Ce, spec.L, **(enc_kw or {}))
        self.nstat = 2 * (self.Ce - 4)
        self.dec = CovDecoder(self.enc.dim + self.nstat, spec.H, self.nc + (0 if exact else 1), spec.Cf if use_fk else 1,
                              dec_dh, drop, fk_mode, spec.Cf_base if (film_base_only and use_fk) else None)
        self.out = PEXOutput(spec, use_anchor, pv_physics, exact)
        self.drop = nn.Dropout(drop)
        self.toa_i, self.anch_i, self.sig, self.mu_pv = spec.toa_unit_idx, spec.anch_idx, spec.sig, spec.pv_mu
        self.info = spec
        self.op_residual = bool(op_residual) and spec.op_idx is not None     # operator forecast = base level of every component
        self.op_idx = spec.op_idx
        self.cap0 = (spec.pv_mu + spec.sig * spec.pv_max_n) * 1.1
        if lin_skip:
            self.lin = nn.ModuleList([nn.Linear(spec.L, spec.H) for _ in range(self.nc)])
            for l in self.lin: nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
            self.pv_bias = nn.Parameter(torch.tensor(0.4))

    def forward(self, x, F, Fh):
        Bn = x.shape[0]
        if self.inst_norm:
            xk = x[..., self.keep]; xs = xk[..., : self.Ce - 4]; m = xs.mean(1, keepdim=True); sd = xs.std(1, keepdim=True) + 1e-2
            xe = torch.cat([(xs - m) / sd, xk[..., self.Ce - 4:]], -1)
            stats = torch.cat([m.squeeze(1), sd.squeeze(1)], -1)
        else:
            xe, stats = x[..., self.keep], torch.zeros(Bn, self.nstat, device=x.device)
        z = torch.cat([self.drop(self.enc(xe)), stats], -1)                   # no dropout on the window statistics
        Fin = F
        if self.rel_anchor:
            Fin = prep_cov(F, Fh, x, self.info, *((m, sd) if self.inst_norm else (None, None)))
        if self.fk_mask:
            Fin = Fin.clone(); Fin[..., self.fk_mask] = 0.0
        if not self.use_fk: Fin = torch.zeros_like(F[..., :1])
        r = self.dec(z, Fin)
        if self.lin_skip:
            lins = []
            for c in range(self.nc):
                if c == 1 and self.out.pv_physics:                           # PV: clear-sky-index space
                    pv_mw = x[..., 2] * self.sig + self.mu_pv; toa = Fh[..., self.toa_i]
                    kap = torch.where(toa > 0.03, (pv_mw / (self.cap0 * toa + 1e-3)).clamp(0, 1.5), torch.zeros_like(toa))
                    bias = self.pv_bias
                    if self.op_residual:                                     # logit of the operator's clear-sky fraction
                        tau = F[..., self.toa_i]; opmw = F[..., self.op_idx + c] * self.sig + self.mu_pv
                        ko = torch.where(tau > 0.03, (opmw / (self.cap0 * tau + 1e-3)).clamp(1e-3, 1 - 1e-3), torch.full_like(tau, 0.5))
                        bias = torch.log(ko) - torch.log1p(-ko)
                    lins.append(self.lin[c](kap) + bias)
                else:
                    ch = c + 1; base = x[:, :, ch].mean(1, keepdim=True)     # window mean
                    if self.op_residual: base = F[..., self.op_idx + c]      # operator forecast of the target time (absolute)
                    lins.append(self.lin[c](x[..., ch] - x[:, :, ch].mean(1, keepdim=True)) + base)
            r_lin = torch.stack(lins, -1)
            if self.exact: r = r + r_lin
            else: r = torch.cat([r[..., : self.nc] + r_lin, r[..., self.nc:]], -1)
        if self.exact:
            return self.out(r, F)
        y = self.out(r[..., : self.nc], F)
        return torch.cat([r[..., self.nc:], y[..., 1:]], -1)


# ------------------------------------------------------------------------------------------------ final configuration
FINAL_HPARAMS = dict(          # validation-tuned (Optuna TPE, 20 trials) and frozen; see README
    enc_kw=dict(d=64, layers=3, heads=8, ds=16, drop=0.3, P=16, S=8),
    dec_dh=128, drop=0.3, fk_mode="film", use_wx=True, film_base_only=True,   # analog covariates: linear path only
)
FINAL_TRAIN = dict(lr=5.231991414754663e-4, weight_decay=6.057676269542806e-06, beta=0.5232530776762708,
                   batch_size=64, epochs=20, patience=6, grad_clip=5.0, scheduler="cosine")


def build_pex_ci(spec, **over):
    """PEX-CI with the frozen hyper-parameters; keyword arguments override single options (e.g. use_fk=False)."""
    kw = dict(FINAL_HPARAMS); kw.update(over)
    return PEX(spec, **kw)


# ------------------------------------------------------------------------------------------------ losses
def pex_loss(beta=0.5232530776762708, w_nl=1.0):
    """MSE(NL_hat) + beta * MSE(components) on the normalised outputs [B,H,T]."""
    def f(out, Y):
        return w_nl * Fn.mse_loss(out[..., 0], Y[..., 0]) + beta * Fn.mse_loss(out[..., 1:], Y[..., 1:])
    return f


def pex_loss_soft(beta=0.5, lam=3.0):
    """Control: four free outputs (NL, L, PV, WD) with the soft balance penalty lam * ||NL - (L - PV - WD)||^2."""
    def f(out, Y):
        return Fn.mse_loss(out[..., 0], Y[..., 0]) + beta * Fn.mse_loss(out[..., 1:], Y[..., 1:]) \
            + lam * ((out[..., 0] - (out[..., 1] - out[..., 2:].sum(-1))) ** 2).mean()
    return f


# ------------------------------------------------------------------------------------------------ conformal intervals
def conformal_quantiles(pred_val, true_val, levels):
    """Per-horizon empirical quantiles of the validation residuals of the net-load forecast (arrays [Nv,H])."""
    import numpy as np
    return np.quantile(true_val - pred_val, levels, axis=0)           # [len(levels), H]


def conformal_interval(pred_test, qv_lo, qv_hi):
    """Split-conformal interval NL_hat + Q_{alpha/2}(e_h), NL_hat + Q_{1-alpha/2}(e_h) for test forecasts [N,H]."""
    return pred_test + qv_lo, pred_test + qv_hi
