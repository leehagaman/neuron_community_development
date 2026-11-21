"""Measure the "an -> vowel-initial next word" behavior across Pythia checkpoints.

For contexts that end in " an" (and, as a control, " a"), we look at the
next-token distribution restricted to word-start tokens (tokens beginning with
a space followed by a letter) and compute the fraction of that mass on
vowel-initial tokens (a, e, i, o, u). The behavior is the gap between the two.

Vowel-initial is defined by spelling, so "an hour" / "a university" count as
noise. That's fine for a first look.
"""

import argparse
import os
import random
import re
from pathlib import Path

# A stale HF token in ~/.cache/huggingface makes public downloads 401; don't send it.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parent.parent
CTX_LEN = 32


def cached_steps(size: str) -> list[int]:
    """Training steps of EleutherAI/pythia-{size} already in the local HF cache."""
    from huggingface_hub.constants import HF_HUB_CACHE

    refs = Path(HF_HUB_CACHE) / f"models--EleutherAI--pythia-{size}" / "refs"
    if not refs.exists():
        return []
    return sorted(int(m.group(1)) for f in refs.iterdir() if (m := re.fullmatch(r"step(\d+)", f.name)))


def build_contexts(tok, n_per_article: int, seed: int = 0):
    """Return {' an': LongTensor[N, CTX_LEN], ' a': ...} contexts ending at the article."""
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    ids = {art: tok.encode(art)[0] for art in (" an", " a")}
    found = {art: [] for art in ids}
    for row in ds:
        toks = tok.encode(row["text"])
        for i in range(CTX_LEN - 1, len(toks) - 1):
            for art, tid in ids.items():
                if toks[i] == tid:
                    found[art].append(toks[i - CTX_LEN + 1 : i + 1])
    rng = random.Random(seed)
    out = {}
    for art, ctxs in found.items():
        rng.shuffle(ctxs)
        out[art] = torch.tensor(ctxs[:n_per_article])
        print(f"{art!r}: {len(ctxs)} occurrences, using {len(out[art])}")
    return out


def vocab_masks(tok, vocab_size: int):
    vowel = torch.zeros(vocab_size, dtype=torch.bool)
    cons = torch.zeros(vocab_size, dtype=torch.bool)
    for i in range(len(tok)):
        s = tok.decode([i])
        if len(s) >= 2 and s[0] == " " and s[1].isascii() and s[1].isalpha():
            (vowel if s[1].lower() in "aeiou" else cons)[i] = True
    return vowel, cons


@torch.no_grad()
def measure(model, contexts, vowel, cons, batch_size=256):
    res = {}
    vowel, cons = vowel.to(model.device), cons.to(model.device)
    for art, ctx in contexts.items():
        fracs, top1_vowel = [], []
        for b in range(0, len(ctx), batch_size):
            x = ctx[b : b + batch_size].to(model.device)
            probs = model(x).logits[:, -1].float().softmax(-1)
            v = probs[:, vowel].sum(-1)
            c = probs[:, cons].sum(-1)
            fracs.append((v / (v + c)).cpu())
            # top-1 among word-start tokens
            ws = probs.masked_fill(~(vowel | cons), 0).argmax(-1)
            top1_vowel.append(vowel[ws].cpu())
        name = art.strip()
        res[f"vowel_frac_{name}"] = torch.cat(fracs).mean().item()
        res[f"top1_vowel_{name}"] = torch.cat(top1_vowel).float().mean().item()
    res["gap"] = res["vowel_frac_an"] - res["vowel_frac_a"]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["160m", "410m", "1b"])
    p.add_argument("--steps", nargs="+", type=int, default=None,
                   help="default: every checkpoint already in the local HF cache")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--out", default=str(ROOT / "results" / "behavior.csv"))
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    contexts = build_contexts(tok, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["model"], r["step"]) for r in rows}

    for size in args.models:
        masks = None
        for step in tqdm(args.steps or cached_steps(size), desc=f"pythia-{size}"):
            if (size, step) in done:
                continue
            model = GPTNeoXForCausalLM.from_pretrained(
                f"EleutherAI/pythia-{size}", revision=f"step{step}", torch_dtype=torch.float32
            ).cuda().eval()
            if masks is None:
                masks = vocab_masks(tok, model.config.vocab_size)
            r = measure(model, contexts, *masks)
            r.update(model=size, step=step)
            rows.append(r)
            print(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
