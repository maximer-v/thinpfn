"""Pre-training on the synthetic prior. Every optimiser step sees `batch_tables` tables of a shared row count drawn by the
prior workers, in micro-batches; the loss is the cross-entropy on query rows. Logged every `log_every` steps: train loss,
the loss on a fixed held-out synthetic batch (and the class-prior loss it must beat), throughput, temperatures and host
memory. Checkpoints every `ckpt_every` steps (resumable from <run>/last.pt), model snapshots every `snap_every`; the last
`lawa_k` snapshots are averaged into <run>/lawa.pt at the end. Optional development AUC on prepared books every `dev_every`.

The released model: python -m thinpfn.train --run thinpfn_v1 --config thinpfn_v1 --steps 16000 --batch-tables 64 --lr 5e-4 --warmup 400
(about 6 hours on one GB10 at 49 tables/s; the run is bound by prior generation on the CPU, use --workers to match your cores).

Reproducibility: --seed seeds the model initialisation and the prior workers (worker w uses seed*1000+w). A resumed run
restores model and optimiser state but the prior workers restart their streams from the beginning, so a resumed run does
not see the same tables as an uninterrupted one; the released weights come from one uninterrupted run.

Safety: a thermal guard pauses when the GPU or the hottest host thermal zone exceeds the limits, and a memory guard saves and
exits when host available memory drops under --min-avail-gb (on unified-memory machines a falling reading does not recover by
waiting). Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the launcher."""
from __future__ import annotations

import argparse, json, math, os, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .model import CONFIGS, ThinPFN
from .priors import mixture

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "runs"


class PriorStream(IterableDataset):
    def __init__(self, mixture_name, micro, n_range, d_max, seed, filter_tables=True):
        self.mixture_name, self.micro, self.n_range, self.d_max, self.seed, self.filter = mixture_name, micro, n_range, d_max, seed, filter_tables

    def __iter__(self):
        info = torch.utils.data.get_worker_info(); wid = info.id if info else 0
        pm = mixture.PriorMixture(self.mixture_name, n_range=self.n_range, d_max=self.d_max, filter_tables=self.filter, seed=self.seed * 1000 + wid)
        while True:
            yield mixture.collate(pm.batch(self.micro), d_max=self.d_max)


def to_device(b, device):
    return {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in b.items()}


def step_loss(model, b, cat_flags: bool):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ic = b["is_categorical"] if cat_flags else torch.zeros_like(b["is_categorical"])
        logits = model(b["X"], b["y"], b["context"], b["feature_mask"], ic, b["cohort"], b["tags"])
    q = ~b["context"]; return F.cross_entropy(logits[q].float(), b["y"][q])


def class_prior_loss(b):
    tot, cnt = 0.0, 0
    for i in range(b["y"].shape[0]):
        c = b["context"][i]; p = b["y"][i][c].float().mean().clamp(1e-4, 1 - 1e-4); yq = b["y"][i][~c].float()
        tot += (-(yq * p.log() + (1 - yq) * (1 - p).log())).sum().item(); cnt += len(yq)
    return tot / max(cnt, 1)


def gpu_temperature():
    try: return float(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0])
    except Exception: return float("nan")


def host_temperature():
    """Hottest /sys/class/thermal zone (NaN when unreadable)."""
    vals = []
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try: vals.append(int(p.read_text().strip()) / 1000.0)
        except Exception: pass
    return max(vals) if vals else float("nan")


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1e6
    except Exception: pass
    return float("nan")


def guards(a, step, model, opt, cfg, state_path):
    avail = mem_available_gb()
    if avail == avail and avail < a.min_avail_gb:
        print(f"memory guard: only {avail:.1f} GB available < {a.min_avail_gb}, saving and exiting at step {step}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg.__dict__}, state_path); raise SystemExit(7)   # the guard runs after opt.step(): `step` is complete
    if step % 10 == 0:
        g, s = gpu_temperature(), host_temperature()
        while (g == g and g > a.max_gpu_c) or (s == s and s > a.max_host_c):
            print(f"thermal guard: GPU {g} C (max {a.max_gpu_c}) / host {s} C (max {a.max_host_c}), pausing 60 s", flush=True); time.sleep(60); g, s = gpu_temperature(), host_temperature()


def lr_at(step, total, peak, warmup):
    if step <= warmup: return peak * step / warmup
    p = (step - warmup) / max(1, total - warmup); return 1e-7 + 0.5 * (peak - 1e-7) * (1 + math.cos(math.pi * p))


def dev_eval(model, device, book_names, ns=(200, 500, 1000), seeds=(0, 1), cat_flags=False):
    """Development AUC on the validation splits of prepared books (never the test split)."""
    from sklearn.metrics import roc_auc_score
    from .data import books
    model.eval(); out = {}
    for name in book_names:
        Xtr, ytr = books.load(name, "train"); Xv, yv = books.load(name, "validation")
        if Xtr.shape[1] > model.cfg.max_features: continue
        if len(yv) > 4000: i = np.random.default_rng(0).choice(len(yv), 4000, replace=False); Xv, yv = Xv[i], yv[i]
        for n in ns:
            aucs = [roc_auc_score(yv, model.predict(*books.subsample_context(Xtr, ytr, n, s), Xv, device=device, is_cat=None if cat_flags else False)) for s in seeds]
            out[f"{name}@{n}"] = float(np.mean(aucs))
    model.train(); out["mean"] = float(np.mean(list(out.values()))); return out


def main(a):
    if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""): print("note: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to limit allocator growth", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True; device = torch.device("cuda"); torch.manual_seed(a.seed)
    run_dir = RUNS / a.run; run_dir.mkdir(parents=True, exist_ok=True); log_path = run_dir / "log.jsonl"; cfg = CONFIGS[a.config]
    if a.fresh:
        for p in list(run_dir.glob("step*.pt")) + [run_dir / "last.pt", run_dir / "lawa.pt", log_path]:
            if p.exists(): p.unlink()
    model = ThinPFN(cfg).to(device); print(f"{a.run}: {model.n_params() / 1e6:.1f}M params, mixture {a.mixture}, {a.steps} steps x {a.batch_tables} tables", flush=True)
    json.dump({**vars(a), "params": model.n_params(), "config": cfg.__dict__}, open(run_dir / "run.json", "w"), indent=1)
    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n_, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), fused=True)
    micro = a.micro; accum = a.batch_tables // micro; n_range = tuple(int(v) for v in a.rows.split("-"))
    ds = PriorStream(a.mixture, micro, n_range, cfg.max_features, seed=a.seed, filter_tables=not a.no_filter)
    loader = iter(DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4, persistent_workers=True, pin_memory=True))
    held = mixture.PriorMixture(a.mixture, n_range=n_range, d_max=cfg.max_features, seed=10_000 + a.seed); held_b = [to_device(mixture.collate(held.batch(4, n=1024), cfg.max_features), device) for _ in range(4)]
    held_prior = float(np.mean([class_prior_loss(b) for b in held_b])); dev_books = [x for x in a.dev_books.split(",") if x]
    start = 0; state_path = run_dir / "last.pt"
    if state_path.exists() and not a.fresh:
        st = torch.load(state_path, map_location=device); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]; print(f"resumed at step {start}", flush=True)
    t0 = time.time(); tables_seen = 0; ema = None; model.train()
    for step in range(start + 1, a.steps + 1):
        lr = lr_at(step, a.steps, a.lr, a.warmup)
        for g in opt.param_groups: g["lr"] = lr
        tot = 0.0
        for _ in range(accum):
            b = to_device(next(loader), device); loss = step_loss(model, b, a.cat_flags) / accum; loss.backward(); tot += loss.item(); tables_seen += micro
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item(); opt.step(); opt.zero_grad(set_to_none=True)
        del b, loss; guards(a, step, model, opt, cfg, state_path)
        ema = tot if ema is None else 0.98 * ema + 0.02 * tot
        if step % a.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad(): hl = float(np.mean([step_loss(model, b, a.cat_flags).item() for b in held_b]))
            model.train(); el = time.time() - t0
            rec = dict(step=step, loss=round(tot, 4), loss_ema=round(ema, 4), held_loss=round(hl, 4), held_prior=round(held_prior, 4), lr=lr, grad_norm=round(gn, 3), tables_per_s=round(tables_seen / el, 1),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), gpu_c=gpu_temperature(), host_c=host_temperature(), host_avail_gb=round(mem_available_gb(), 1), elapsed_min=round(el / 60, 1))
            print(json.dumps(rec), flush=True); open(log_path, "a").write(json.dumps(rec) + "\n")
        if step % a.ckpt_every == 0 or step == a.steps: torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "config": cfg.__dict__}, state_path)
        if step % (a.snap_every or a.ckpt_every) == 0 or step == a.steps: torch.save({"model": model.state_dict(), "step": step, "config": cfg.__dict__}, run_dir / f"step{step:06d}.pt")
        if a.dev_every and dev_books and (step % a.dev_every == 0 or step == a.steps):
            rec = dict(step=step, dev=dev_eval(model, device, dev_books, cat_flags=a.cat_flags)); print(json.dumps(rec), flush=True); open(log_path, "a").write(json.dumps(rec) + "\n")
    cks = sorted(p for p in run_dir.glob("step*.pt") if int(p.stem[4:]) <= a.steps)[-a.lawa_k:]
    if cks:   # LAWA: average the last k snapshots
        avg = None
        for p in cks:
            sd = torch.load(p, map_location="cpu")["model"]; avg = {k: v.float().clone() for k, v in sd.items()} if avg is None else {k: avg[k] + sd[k].float() for k in avg}
        avg = {k: v / len(cks) for k, v in avg.items()}; torch.save({"model": avg, "config": cfg.__dict__, "averaged": [p.name for p in cks]}, run_dir / "lawa.pt")
        if dev_books: model.load_state_dict(avg); rec = dict(step="lawa", dev=dev_eval(model, device, dev_books, cat_flags=a.cat_flags)); print(json.dumps(rec), flush=True); open(log_path, "a").write(json.dumps(rec) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--config", default="thinpfn_v1", choices=list(CONFIGS)); ap.add_argument("--mixture", default="default", choices=list(mixture.MIXTURES))
    ap.add_argument("--steps", type=int, default=16000); ap.add_argument("--batch-tables", type=int, default=64); ap.add_argument("--micro", type=int, default=8); ap.add_argument("--rows", default="256-1024")
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--warmup", type=int, default=400); ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-filter", action="store_true"); ap.add_argument("--cat-flags", action="store_true", help="feed categorical flags (the released model did not)")
    ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--ckpt-every", type=int, default=50); ap.add_argument("--snap-every", type=int, default=500); ap.add_argument("--lawa-k", type=int, default=6)
    ap.add_argument("--dev-every", type=int, default=2000); ap.add_argument("--dev-books", default="", help="comma-separated prepared books for a development AUC every --dev-every steps (needs `python -m thinpfn.data.books prepare ...`; empty = synthetic-only run)")
    ap.add_argument("--max-gpu-c", type=float, default=78); ap.add_argument("--max-host-c", type=float, default=90); ap.add_argument("--min-avail-gb", type=float, default=16); ap.add_argument("--fresh", action="store_true")
    main(ap.parse_args())
