"""ThinPFN: a prior-fitted network for small tabular classification problems (own implementation; the public TabPFN and
TabICL papers were references for the design, no code is copied).

Input per table: X [n, d] (NaN allowed, padded columns NaN with feature_mask False), y [n] (used on context rows only),
context [n] bool, cohort [n] in [0, 1]. Three stages:
  1. column embedding: every column is a set of n scalar values; an induced set attention block (inducing points attend
     to the CONTEXT values only, then every value attends to the inducing summary) turns each cell into a vector, so a
     cell's embedding knows the column's distribution without looking at query rows.
  2. row interaction: for every row, attention over its d cell vectors plus 4 CLS tokens (RoPE on the column index as
     symmetry breaking; columns are permuted at train time); the CLS outputs concatenated give the row token.
  3. in-context learning: a transformer over all rows; context rows carry a label embedding and attend to each other,
     query rows attend to the context (and themselves) only; learned thinking rows are appended to the context.
A Time2Vec of the cohort index is added to every row token (the domain channel). Head: 2-way logits on query rows."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PFNConfig:
    d_col: int = 96; col_inducing: int = 64; col_blocks: int = 2; col_heads: int = 4
    row_layers: int = 2; row_heads: int = 4; n_cls: int = 4
    d_icl: int = 256; icl_layers: int = 6; icl_heads: int = 8; ff_mult: int = 2
    n_think: int = 16; max_features: int = 64; dropout: float = 0.0; n_classes: int = 2; residual_decay: float = 0.95
    zero_init_pooled: bool = True   # zero-init the CLS-pooled path so training starts from the raw skip alone
    n_tags: int = 48           # credit field vocabulary size (schema.N_TYPES <= n_tags); 0 disables the tag channel
    raw_skip: bool = True      # TabPFN-v1-style linear map of the (z-scored, NaN-filled) feature vector added to the row token

    @property
    def d_row(self): return self.d_col * self.n_cls


# per-row core (TabPFN v1 style): no column / row stages; the raw skip is the row embedding. Chosen for the smoke stage on
# 8 Nov 2026 because the column / row stages delay the in-context transition by thousands of steps and cost 7x per step.
# Released configuration (thinpfn_v1, 25.6M parameters): a per-row model, no column or row stages, 12 in-context layers of
# width 512 with 24 thinking rows. SMALL is the 3.3M development configuration used for ablations.
THINPFN_V1 = PFNConfig(col_blocks=0, row_layers=0, d_icl=512, icl_layers=12, icl_heads=8, n_think=24)
SMALL = PFNConfig(col_blocks=0, row_layers=0, n_think=16)
STAGED = PFNConfig(d_col=128, col_inducing=128, col_blocks=3, col_heads=8, row_layers=3, row_heads=8, d_icl=512, icl_layers=12, icl_heads=8, n_think=24)   # column and row stages on (slower; not released)
CONFIGS = {"thinpfn_v1": THINPFN_V1, "small": SMALL, "staged": STAGED}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class MHA(nn.Module):
    """Multi-head attention with separate query / key-value inputs and an optional boolean mask [B, Lq, Lk] (True = attend)."""
    def __init__(self, d, heads, dropout=0.0):
        super().__init__(); assert d % heads == 0; self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d, bias=False); self.kv = nn.Linear(d, 2 * d, bias=False); self.o = nn.Linear(d, d, bias=False); self.dropout = dropout
    def forward(self, xq, xkv, mask=None, rope=None):
        B, Lq, D = xq.shape; Lk = xkv.shape[1]
        q = self.q(xq).view(B, Lq, self.h, self.dk).transpose(1, 2); k, v = self.kv(xkv).view(B, Lk, 2, self.h, self.dk).permute(2, 0, 3, 1, 4)
        if rope is not None: q, k = rope(q), rope(k)
        am = mask[:, None] if mask is not None else None   # [B, 1, Lq, Lk]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, Lq, D))


class FF(nn.Module):
    def __init__(self, d, mult, dropout=0.0):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d, d))
    def forward(self, x): return self.net(x)


class Block(nn.Module):
    """Pre-norm self-attention block with residual scaling."""
    def __init__(self, d, heads, mult, scale=1.0, dropout=0.0):
        super().__init__(); self.n1, self.n2 = RMSNorm(d), RMSNorm(d); self.attn = MHA(d, heads, dropout); self.ff = FF(d, mult, dropout); self.scale = scale
    def forward(self, x, mask=None, rope=None):
        h = self.n1(x); x = x + self.scale * self.attn(h, h, mask, rope); return x + self.scale * self.ff(self.n2(x))


class ISAB(nn.Module):
    """Induced set attention: inducing points attend to the context values; all values then attend to the inducing summary."""
    def __init__(self, d, heads, m):
        super().__init__(); self.I = nn.Parameter(torch.randn(m, d) * 0.02); self.a1, self.a2 = MHA(d, heads), MHA(d, heads)
        self.n1, self.n2, self.n3 = RMSNorm(d), RMSNorm(d), RMSNorm(d); self.ff = FF(d, 2)
    def forward(self, x, ctx_mask):
        # x [S, n, d]; ctx_mask [S, n] True for context rows: inducing points see context values only
        S, n, d = x.shape; I = self.I[None].expand(S, -1, -1)
        H = I + self.a1(self.n1(I), self.n1(x), ctx_mask[:, None, :].expand(-1, I.shape[1], -1))
        x = x + self.a2(self.n2(x), self.n2(H)); return x + self.ff(self.n3(x))


class RoPE:
    def __init__(self, dk, base=10000.0):
        self.dk, self.base = dk, base; self.cache = {}
    def __call__(self, x):  # x [B, h, L, dk]
        L = x.shape[2]; key = (L, x.device, x.dtype)
        if key not in self.cache:
            inv = 1.0 / (self.base ** (torch.arange(0, self.dk, 2, device=x.device).float() / self.dk)); t = torch.arange(L, device=x.device).float()
            f = torch.outer(t, inv); self.cache[key] = (f.cos().to(x.dtype), f.sin().to(x.dtype))
        cos, sin = self.cache[key]; x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Time2Vec(nn.Module):
    def __init__(self, d, k=8):
        super().__init__(); self.lin = nn.Linear(1, 1); self.per = nn.Linear(1, k - 1); self.out = nn.Linear(k, d)
    def forward(self, c):  # c [B, n]
        c = c[..., None]; return self.out(torch.cat([self.lin(c), torch.sin(self.per(c))], -1))


class ThinPFN(nn.Module):
    def __init__(self, cfg: "PFNConfig" = None):
        cfg = cfg or THINPFN_V1
        super().__init__(); self.cfg = cfg; c = cfg
        # 1. column embedding: cell scalar -> d_col, with NaN and categorical indicators, then ISAB blocks over rows
        self.cell_in = nn.Sequential(nn.Linear(3, c.d_col, bias=False), nn.GELU(), nn.Linear(c.d_col, c.d_col, bias=False))   # no biases: a bias shared by every cell sums its gradient coherently over ~1e5 cells and dominates the global norm
        self.col_blocks = nn.ModuleList([ISAB(c.d_col, c.col_heads, c.col_inducing) for _ in range(c.col_blocks)])
        # 2. row interaction over cells + CLS tokens
        self.cls = nn.Parameter(torch.randn(c.n_cls, c.d_col) * 0.02); self.rope = RoPE(c.d_col // c.row_heads)
        self.row_blocks = nn.ModuleList([Block(c.d_col, c.row_heads, c.ff_mult, c.residual_decay ** i, c.dropout) for i in range(c.row_layers)])
        self.row_norm = RMSNorm(c.d_col)
        # 3. in-context stage
        self.row_proj = nn.Linear(c.d_row, c.d_icl); self.raw_proj = nn.Linear(3 * c.max_features, c.d_icl, bias=False) if c.raw_skip else None
        # tag channel: slot-invariant sum over columns of a value embedding modulated by the field-type embedding (FiLM); UNKNOWN = id 0
        if c.n_tags:
            self.tag_val = nn.Sequential(nn.Linear(3, 64, bias=False), nn.GELU(), nn.Linear(64, c.d_icl, bias=False)); self.tag_scale = nn.Embedding(c.n_tags, c.d_icl); self.tag_shift = nn.Embedding(c.n_tags, c.d_icl)
        self.label_emb = nn.Embedding(c.n_classes + 1, c.d_icl)   # index n_classes = "query, unknown"
        self.cohort = Time2Vec(c.d_icl); self.think = nn.Parameter(torch.randn(c.n_think, c.d_icl) * 0.02)
        self.icl_blocks = nn.ModuleList([Block(c.d_icl, c.icl_heads, c.ff_mult, c.residual_decay ** i, c.dropout) for i in range(c.icl_layers)])
        self.icl_norm = RMSNorm(c.d_icl); self.head = nn.Linear(c.d_icl, c.n_classes)
        self.apply(self._init)
        # the cell MLP reads a 3-wide input: give it unit-scale outputs (with the 0.02 init the cell embeddings had std 0.06)
        for lin in (self.cell_in[0], self.cell_in[2]):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
        self.pack_columns = True
        if c.n_tags:   # the tag channel starts silent and grows in (after the global init, which would otherwise overwrite this)
            nn.init.ones_(self.tag_scale.weight); nn.init.zeros_(self.tag_shift.weight); nn.init.zeros_(self.tag_val[2].weight)
        if c.raw_skip and c.zero_init_pooled:   # start as a per-row (TabPFN v1 style) model; the column / row stages grow in through training
            nn.init.zeros_(self.row_proj.weight); nn.init.zeros_(self.row_proj.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02);
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.trunc_normal_(m.weight, std=0.02)

    def n_params(self): return sum(p.numel() for p in self.parameters())

    def embed_columns(self, X, feature_mask, is_cat, context):
        B, n, d = X.shape; nan = torch.isnan(X); x = torch.nan_to_num(X, nan=0.0)
        cells = torch.stack([x, nan.to(x.dtype), is_cat[:, None, :].expand(-1, n, -1).to(x.dtype)], -1)  # [B, n, d, 3]
        h = self.cell_in(cells).permute(0, 2, 1, 3)                                                   # [B, d, n, d_col]
        if self.pack_columns:   # run the set attention on the real columns only (padding columns are masked out downstream)
            bi, ci = torch.nonzero(feature_mask, as_tuple=True); packed = h[bi, ci].contiguous(); ctx = context[bi]
            for blk in self.col_blocks: packed = blk(packed, ctx)
            out = torch.zeros_like(h); out[bi, ci] = packed.to(out.dtype); return out.permute(0, 2, 1, 3)
        h = h.reshape(B * d, n, self.cfg.d_col); ctx = context[:, None, :].expand(-1, d, -1).reshape(B * d, n)
        for blk in self.col_blocks: h = blk(h, ctx)
        return h.view(B, d, n, -1).permute(0, 2, 1, 3)                                              # [B, n, d, d_col]

    def embed_rows(self, cells, feature_mask):
        B, n, d, dc = cells.shape; c = self.cfg
        tok = torch.cat([self.cls[None, None].expand(B, n, -1, -1), cells], 2).reshape(B * n, c.n_cls + d, dc)
        keep = torch.cat([torch.ones(B, c.n_cls, dtype=torch.bool, device=cells.device), feature_mask], 1)   # [B, n_cls + d]
        mask = keep[:, None, :].expand(-1, c.n_cls + d, -1)                                                 # [B, L, L] keys masked
        mask = mask[:, None].expand(-1, n, -1, -1).reshape(B * n, c.n_cls + d, c.n_cls + d)
        for blk in self.row_blocks: tok = blk(tok, mask, self.rope)
        tok = self.row_norm(tok[:, :c.n_cls]).reshape(B, n, c.d_row)
        return self.row_proj(tok)                                                                            # [B, n, d_icl]

    def forward(self, X, y, context, feature_mask, is_cat, cohort=None, tags=None):
        """Returns logits [B, n, n_classes] for every row (loss is taken on query rows only). tags [B, d] field-type ids (optional)."""
        B, n, d = X.shape; c = self.cfg
        rows = self.embed_rows(self.embed_columns(X, feature_mask, is_cat, context), feature_mask)
        cells3 = torch.stack([torch.nan_to_num(X, nan=0.0), torch.isnan(X).to(X.dtype), is_cat[:, None, :].expand(-1, n, -1).to(X.dtype)], -1)   # [B, n, d, 3]
        if self.raw_proj is not None:   # raw skip: features in their (random) slots plus NaN indicators, so row comparison needs no learned pooling
            raw = torch.cat([torch.nan_to_num(X, nan=0.0), torch.isnan(X).to(X.dtype), is_cat[:, None, :].expand(-1, n, -1).to(X.dtype)], -1).to(rows.dtype); rows = rows + self.raw_proj(raw)
        if c.n_tags and tags is not None:   # tag channel: knowledge transfers across lenders through field types, not slot positions
            v = self.tag_val(cells3.to(rows.dtype)) * self.tag_scale(tags)[:, None] + self.tag_shift(tags)[:, None]            # [B, n, d, d_icl]
            v = v * feature_mask[:, None, :, None].to(v.dtype); rows = rows + v.sum(2) / math.sqrt(max(1.0, float(d)))
        lab = torch.where(context, y.clamp(0, c.n_classes - 1), torch.full_like(y, c.n_classes)); rows = rows + self.label_emb(lab)
        if cohort is not None: rows = rows + self.cohort(cohort.to(rows.dtype))
        T = c.n_think; h = torch.cat([self.think[None].expand(B, -1, -1), rows], 1)                       # [B, T + n, d_icl]
        key_ok = torch.cat([torch.ones(B, T, dtype=torch.bool, device=X.device), context], 1)               # keys: thinking rows + context rows
        mask = key_ok[:, None, :].expand(-1, T + n, -1).clone()
        idx = torch.arange(T + n, device=X.device); mask[:, idx, idx] = True                                 # every row sees itself
        for blk in self.icl_blocks: h = blk(h, mask)
        return self.head(self.icl_norm(h[:, T:]))

    @torch.no_grad()
    def predict(self, X_ctx, y_ctx, X_q, cohort_ctx=None, cohort_q=None, is_cat=None, device=None, batch_rows=4096, tags=None, preprocess="z", n_perm=0, seed=0):
        """Numpy convenience: log-odds of class 1 for the query rows given one context. Numeric features are standardised on
        the context, categoricals left as codes. preprocess "z": z-score clipped at +-4 sigma (the prior's own standardisation);
        "qnorm": quantile-normal transform fitted on the context (recommended: real columns are heavy-tailed and discrete,
        and the z-score squashes them while the prior's inputs are Gaussian-shaped). n_perm > 0: average the log-odds over
        n_perm random column permutations (the row map is slot-wise, so the model is not permutation invariant)."""
        import numpy as np
        device = torch.device(device) if device is not None else next(self.parameters()).device
        X_ctx = np.asarray(X_ctx); X_q = np.asarray(X_q); y_ctx = np.asarray(y_ctx)
        if X_ctx.ndim != 2 or X_q.ndim != 2 or X_q.shape[1] != X_ctx.shape[1]: raise ValueError("X_ctx and X_q must be 2-D with the same number of columns")
        if len(y_ctx) != len(X_ctx): raise ValueError("y_ctx must have one label per context row")
        d = X_ctx.shape[1]
        if d > self.cfg.max_features: raise ValueError(f"at most {self.cfg.max_features} features are supported (got {d})")
        if len(X_q) == 0: return np.zeros(0, np.float32)
        if is_cat is None:   # heuristic flag: <= 12 distinct integer values; the released weights were trained without flags, pass is_cat=False
            from .priors.contract import looks_categorical
            is_cat = np.array([looks_categorical(X_ctx[:, j]) for j in range(d)])
        elif is_cat is False: is_cat = np.zeros(d, bool)
        else: is_cat = np.asarray(is_cat, bool)
        Xc = X_ctx.astype(np.float32).copy(); Xq = X_q.astype(np.float32).copy()
        for j in range(d):
            if is_cat[j]: continue
            col = Xc[:, j]; ok = np.isfinite(col)
            if preprocess == "qnorm" and ok.sum() >= 2:
                from scipy.stats import norm
                ref = np.sort(col[ok])
                def tr(v):
                    o = v.copy(); m_ = np.isfinite(v); r = 0.5 * (np.searchsorted(ref, v[m_], "left") + np.searchsorted(ref, v[m_], "right"))
                    o[m_] = norm.ppf(np.clip(r / len(ref), 0.5 / len(ref), 1 - 0.5 / len(ref))); return o
                Xc[:, j] = tr(Xc[:, j]); Xq[:, j] = tr(Xq[:, j]); continue
            m = np.nanmean(col) if ok.any() else 0.0; s = np.nanstd(col) if ok.any() else 1.0; s = s if s > 1e-8 else 1.0
            Xc[:, j] = np.clip((Xc[:, j] - m) / s, -4, 4); Xq[:, j] = np.clip((Xq[:, j] - m) / s, -4, 4)
        if n_perm:
            rng = np.random.default_rng(seed); outs = []
            for _ in range(n_perm):
                p = rng.permutation(d)
                outs.append(self._predict_std(Xc[:, p], y_ctx, Xq[:, p], is_cat[p], None if tags is None else np.asarray(tags)[p], cohort_ctx, cohort_q, device, batch_rows))
            return np.mean(outs, 0)
        return self._predict_std(Xc, y_ctx, Xq, is_cat, tags, cohort_ctx, cohort_q, device, batch_rows)

    @torch.no_grad()
    def _predict_std(self, Xc, y_ctx, Xq, is_cat, tags, cohort_ctx, cohort_q, device, batch_rows):
        import numpy as np
        d = Xc.shape[1]; out = []
        for start in range(0, len(Xq), batch_rows):
            q = Xq[start:start + batch_rows]; n = len(Xc) + len(q)
            X = np.full((1, n, self.cfg.max_features), np.nan, np.float32); X[0, :, :d] = np.concatenate([Xc, q])
            y = np.zeros((1, n), np.int64); y[0, :len(Xc)] = y_ctx; ctx = np.zeros((1, n), bool); ctx[0, :len(Xc)] = True
            fm = np.zeros((1, self.cfg.max_features), bool); fm[0, :d] = True; ic = np.zeros((1, self.cfg.max_features), bool); ic[0, :d] = is_cat
            tg = np.zeros((1, self.cfg.max_features), np.int64)
            if tags is not None: tg[0, :d] = np.asarray(tags, np.int64)
            coh = None
            if cohort_ctx is not None:
                cc = np.concatenate([cohort_ctx, cohort_q[start:start + batch_rows]]).astype(np.float32); lo, hi = cohort_ctx.min(), cohort_ctx.max()
                coh = torch.tensor(((cc - lo) / (hi - lo + 1e-9)).clip(0, 1.5)[None], device=device)
            t = lambda a: torch.tensor(a, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = self.forward(t(X), t(y), t(ctx), t(fm), t(ic), coh, t(tg))[0, len(Xc):].float()
            out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(out)
