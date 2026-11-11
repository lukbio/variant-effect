"""Build the labeled missense dataset from ClinVar, with protein sequences.

Usage:
    python scripts/build_dataset.py --email you@example.org

Reads data/variant_summary.txt.gz (ClinVar's tab-delimited summary) and keeps
single-nucleotide variants that change one amino acid, reviewed to at least
two stars, and classified (likely) pathogenic or (likely) benign. Fetches each
variant's protein (the translation of its RefSeq transcript) from NCBI,
checks that the protein has the reference amino acid at the stated position,
and assigns every gene to one of five folds so all of a gene's variants stay
on the same side of a train/test split.

Writes data/variants.tsv and data/proteins.fasta.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
THREE = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
         "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
         "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
# Two stars or more.
REVIEW = {"criteria provided, multiple submitters, no conflicts", "reviewed by expert panel", "practice guideline"}
LABELS = {"Pathogenic": 1, "Likely pathogenic": 1, "Pathogenic/Likely pathogenic": 1,
          "Benign": 0, "Likely benign": 0, "Benign/Likely benign": 0}
NAME = re.compile(r"^(NM_\d+\.\d+)\(([^)]+)\):c\.\S+ \(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)$")
FOLDS = 5


def eutil(endpoint: str, email: str, **params) -> bytes:
    data = urlencode({**params, "email": email, "tool": "variant-effect"}).encode()
    for attempt in range(5):
        try:
            with urlopen(Request(f"{EUTILS}/{endpoint}.fcgi", data=data), timeout=300) as r:
                return r.read()
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  retrying after {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def read_clinvar(path: Path) -> pd.DataFrame:
    cols = ["Type", "Name", "GeneSymbol", "ClinicalSignificance", "ReviewStatus", "Assembly", "VariationID"]
    df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str, compression="gzip")
    df = df[(df.Assembly == "GRCh38") & (df.Type == "single nucleotide variant")]
    df = df[df.ReviewStatus.isin(REVIEW) & df.ClinicalSignificance.isin(LABELS)]
    m = df.Name.str.extract(NAME)
    m.columns = ["transcript", "gene", "ref3", "pos", "alt3"]
    df = pd.concat([df.reset_index(drop=True), m.reset_index(drop=True)], axis=1).dropna(subset=["transcript"])
    df = df[df.ref3.isin(THREE) & df.alt3.isin(THREE) & (df.ref3 != df.alt3)]  # missense only
    out = pd.DataFrame({
        "variation_id": df.VariationID,
        "gene": df.gene,
        "transcript": df.transcript,
        "pos": df.pos.astype(int),
        "ref": df.ref3.map(THREE),
        "alt": df.alt3.map(THREE),
        "label": df.ClinicalSignificance.map(LABELS),
        "review": df.ReviewStatus,
    })
    return out.drop_duplicates(["transcript", "pos", "ref", "alt"]).reset_index(drop=True)


def fetch_proteins(transcripts: list[str], email: str, cache: Path) -> dict[str, str]:
    """{transcript: protein} from each transcript's CDS translation (fasta_cds_aa)."""
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for i in range(0, len(transcripts), 200):
        chunk = transcripts[i:i + 200]
        f = cache / f"cds_aa_{hashlib.md5(','.join(chunk).encode()).hexdigest()[:12]}.fa"
        if not f.exists() or f.stat().st_size == 0:
            print(f"  fetching proteins {i + 1}-{i + len(chunk)} of {len(transcripts)}", file=sys.stderr)
            f.write_bytes(eutil("efetch", email, db="nuccore", id=",".join(chunk), rettype="fasta_cds_aa", retmode="text"))
            time.sleep(0.4)
        for block in f.read_text().split("\n>"):
            header, _, seq = block.lstrip(">").partition("\n")
            m = re.match(r"lcl\|(NM_\d+\.\d+)_prot_", header)
            if m and m.group(1) not in out:  # first CDS of the transcript
                out[m.group(1)] = re.sub(r"\s+", "", seq).rstrip("*")
    return out


def fold_of(gene: str) -> int:
    """Stable gene -> fold assignment (same on every run and machine)."""
    return int(hashlib.sha1(gene.encode()).hexdigest(), 16) % FOLDS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True)
    ap.add_argument("--clinvar", default="data/variant_summary.txt.gz")
    args = ap.parse_args()

    df = read_clinvar(Path(args.clinvar))
    print(f"{len(df)} reviewed missense variants in {df.gene.nunique()} genes", file=sys.stderr)

    proteins = fetch_proteins(sorted(df.transcript.unique()), args.email, Path("data/cache"))
    seq = df.transcript.map(proteins)
    ok = pd.Series([isinstance(s, str) and p <= len(s) and s[p - 1] == r
                    for s, p, r in zip(seq, df.pos, df.ref)], index=df.index)
    dropped = (~ok).sum()
    df = df[ok].reset_index(drop=True)
    print(f"dropped {dropped} whose reference amino acid didn't match the protein", file=sys.stderr)

    df["length"] = df.transcript.map(lambda t: len(proteins[t]))
    df["fold"] = df.gene.map(fold_of)
    df.to_csv("data/variants.tsv", sep="\t", index=False)
    with open("data/proteins.fasta", "w") as fh:
        for t in sorted(df.transcript.unique()):
            fh.write(f">{t}\n{proteins[t]}\n")

    print(f"wrote {len(df)} variants, {df.transcript.nunique()} proteins, {df.gene.nunique()} genes", file=sys.stderr)
    print(df.groupby("fold").agg(variants=("label", "size"), pathogenic=("label", "sum"), genes=("gene", "nunique")),
          file=sys.stderr)


if __name__ == "__main__":
    main()
