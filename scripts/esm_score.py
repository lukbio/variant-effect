"""Score every variant with ESM-2, and save its embeddings for later models.

Usage:
    python scripts/esm_score.py [--model facebook/esm2_t30_150M_UR50D]

For each protein, one forward pass on the unmasked sequence gives log
probabilities for all 20 amino acids at every position ("wild-type
marginals"). A variant's zero-shot score is

    log P(alt | protein) - log P(ref | protein)

at its position: negative when the model finds the mutant amino acid less
likely than the original, which tends to mean harmful.

ESM-2 reads at most 1,022 residues, so longer proteins are cut into
overlapping windows and each position is scored from the window where it sits
furthest from an edge.

Also saves the last-layer embedding at each variant's position (for the
supervised models). Writes results/esm_<model>.parquet (scores) and
data/emb_<model>.npy (embeddings, one row per variant, same order).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

MAX_RES = 1022
STRIDE = 511


def windows(length: int) -> list[tuple[int, int]]:
    """(start, end) residue windows covering a protein of this length."""
    if length <= MAX_RES:
        return [(0, length)]
    starts = list(range(0, length - MAX_RES, STRIDE)) + [length - MAX_RES]
    return [(s, s + MAX_RES) for s in starts]


def best_window(pos: int, wins: list[tuple[int, int]]) -> int:
    """Index of the window where 0-based position pos is furthest from an edge."""
    return max(range(len(wins)), key=lambda i: min(pos - wins[i][0], wins[i][1] - 1 - pos)
               if wins[i][0] <= pos < wins[i][1] else -1)


def read_fasta(path: str) -> dict[str, str]:
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            out[name] = ""
        elif name:
            out[name] += line
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    ap.add_argument("--limit", type=int, default=0, help="only the first N proteins (for a quick test)")
    ap.add_argument("--checkpoint-every", type=int, default=250,
                    help="save progress every N proteins; a rerun picks up from the last save")
    args = ap.parse_args()
    short = args.model.split("/")[-1]

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = EsmForMaskedLM.from_pretrained(args.model).eval()
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in "ACDEFGHIKLMNPQRSTVWY"}

    variants = pd.read_csv("data/variants.tsv", sep="\t")
    proteins = read_fasta("data/proteins.fasta")
    groups = variants.groupby("transcript").groups
    names = sorted(groups)
    if args.limit:
        names = names[: args.limit]

    scores = np.full(len(variants), np.nan, dtype=np.float32)
    emb = None
    ckpt, first = Path(f"data/ckpt_{short}.npz"), 0
    if ckpt.exists():
        saved = np.load(ckpt)
        if len(saved["scores"]) == len(variants):
            scores, emb, first = saved["scores"], saved["emb"], int(saved["done"])
            print(f"resuming after {first} proteins", flush=True)
    t0 = time.time()
    for k, name in enumerate(names):
        if k < first:
            continue
        seq = proteins[name]
        idx = list(groups[name])
        wins = windows(len(seq))
        by_win: dict[int, list[int]] = {}
        for i in idx:
            by_win.setdefault(best_window(int(variants.pos[i]) - 1, wins), []).append(i)
        for w, rows in by_win.items():
            start, end = wins[w]
            enc = tok(seq[start:end], return_tensors="pt")
            out = model(**enc, output_hidden_states=True)
            logp = torch.log_softmax(out.logits[0], dim=-1)  # token 0 is <cls>
            hidden = out.hidden_states[-1][0]
            if emb is None:
                emb = np.zeros((len(variants), hidden.shape[-1]), dtype=np.float16)
            for i in rows:
                t = int(variants.pos[i]) - start  # 1-based residue -> token index (after <cls>)
                scores[i] = (logp[t, aa_ids[variants.alt[i]]] - logp[t, aa_ids[variants.ref[i]]]).item()
                emb[i] = hidden[t].numpy().astype(np.float16)
        if (k + 1) % args.checkpoint_every == 0:
            tmp = ckpt.with_suffix(".tmp.npz")
            np.savez(tmp, scores=scores, emb=emb, done=k + 1)
            os.replace(tmp, ckpt)
        if (k + 1) % 100 == 0:
            rate = (time.time() - t0) / (k + 1 - first)
            print(f"{k + 1}/{len(names)} proteins, {rate:.2f} s each, ~{rate * (len(names) - k - 1) / 60:.0f} min left",
                  flush=True)

    Path("results").mkdir(exist_ok=True)
    out = variants[["variation_id", "gene", "transcript", "pos", "ref", "alt", "label", "fold"]].copy()
    out["esm_llr"] = scores
    out.to_parquet(f"results/esm_{short}.parquet", index=False)
    np.save(f"data/emb_{short}.npy", emb)
    ckpt.unlink(missing_ok=True)
    print(f"scored {np.isfinite(scores).sum()} of {len(scores)} variants in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
