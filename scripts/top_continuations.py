"""Track the corpus' top-3 continuations of ' a' and ' an' across training.

Per checkpoint, article, and word: mean model probability (with bootstrap 95% CI
over contexts), median per-context rank, and share of contexts where it is top-1.
Uses every checkpoint in the local HF cache; skips (model, step) pairs already in
results/top_continuations.csv. Plot with plot_top_continuations.py.
"""

import collections
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from behavior_over_training import build_contexts, cached_steps

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["160m", "410m", "1b"]
TOP_N = 3
N_BOOT = 1000


def main(*models):
    models = models or MODELS
    out = ROOT / "results" / "top_continuations.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["model"], r["step"]) for r in rows}
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    contexts = build_contexts(tok, 2000)
    ids = {art: tok.encode(art)[0] for art in contexts}
    counts = {art: collections.Counter() for art in contexts}
    for row in load_dataset("NeelNanda/pile-10k", split="train"):
        t = tok.encode(row["text"])
        for i in range(len(t) - 1):
            for art, tid in ids.items():
                if t[i] == tid:
                    counts[art][t[i + 1]] += 1
    targets = {}
    for art, c in counts.items():
        tot = sum(c.values())
        targets[art] = [(t, n / tot) for t, n in c.most_common(TOP_N)]
        print(art, [(tok.decode([t]), round(p, 4)) for t, p in targets[art]])

    gen = torch.Generator().manual_seed(0)
    todo = [(m, s) for m in models for s in cached_steps(m) if (m, s) not in done]
    for size, step in tqdm(todo):
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{size}", revision=f"step{step}").cuda().eval()
        for art, ctx in contexts.items():
            with torch.no_grad():
                p = torch.cat([model(ctx[b:b + 250].cuda()).logits[:, -1].float().softmax(-1)
                               for b in range(0, len(ctx), 250)])
            top1 = p.argmax(-1)
            boot = torch.randint(0, len(ctx), (N_BOOT, len(ctx)), generator=gen).cuda()
            for t, corpus_p in targets[art]:
                pt = p[:, t]
                rank = (p > pt[:, None]).sum(-1) + 1
                bm = pt[boot].mean(-1)
                rows.append({
                    "model": size, "step": step, "article": art.strip(), "token": tok.decode([t]),
                    "corpus_p": corpus_p, "model_p": pt.mean().item(),
                    "model_p_lo": bm.quantile(0.025).item(), "model_p_hi": bm.quantile(0.975).item(),
                    "median_rank": rank.float().median().item(),
                    "top1_share": (top1 == t).float().mean().item(),
                })
        del model
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
