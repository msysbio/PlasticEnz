#!/usr/bin/env python3
"""
Annotate a PlasticEnz Summary_table.tsv with SignalP 6 predictions.

- Reads Protein Name / Protein Sequence
- Runs signalp6 (txt output)
- Parses columns: ID, Prediction, subtype probs, CS Position
- Merges: signalp6_pred, signalp6_type, signalp6_cs, signalp6_prob
- Keeps temp workspace optionally and copies raw SignalP table next to output.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from io import StringIO
from pathlib import Path
from typing import Tuple, List

import pandas as pd

# Fixed PlasticEnz columns
REQ_ID_COL = "Protein Name"
REQ_SEQ_COL = "Protein Sequence"

# Subtype columns in SignalP6 txt
SUBTYPE_COLS: List[str] = [
    "SP(Sec/SPI)",
    "LIPO(Sec/SPII)",
    "TAT(Tat/SPI)",
    "TATLIPO(Tat/SPII)",
    "PILIN(Sec/SPIII)",
]


def write_fasta(df: pd.DataFrame, fasta_path: Path) -> None:
    """Write a FASTA using Protein Name as header and Protein Sequence as sequence."""
    with open(fasta_path, "w") as fh:
        for _, r in df.iterrows():
            sid = str(r[REQ_ID_COL])
            seq = str(r[REQ_SEQ_COL]).replace(" ", "").replace("\n", "").replace("*", "")
            if not seq or seq.lower() == "nan":
                continue
            fh.write(f">{sid}\n{seq}\n")


def find_txt_table(outdir: Path) -> Path:
    """Find SignalP6 txt result file, preferring common names."""
    preferred = ["prediction_results.txt", "results.txt", "signalp6_results.txt"]
    for name in preferred:
        p = outdir / name
        if p.exists():
            return p
    txts = sorted(outdir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No SignalP6 .txt output found in {outdir}")
    return txts[0]


def _read_signalp_txt(txt_path: Path) -> pd.DataFrame:
    """
    Read SignalP6 txt which has a commented header line (starting with '#').
    We take the LAST commented header that contains tabs, then the rest as data.
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    header = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#") and "\t" in line:
            header = line.lstrip("#").strip()
            data_start = i + 1
    if header is None:
        # Fallback: let pandas infer, skipping comment lines
        return pd.read_csv(txt_path, sep="\t", dtype=str, comment="#")

    cols = [c.strip() for c in header.split("\t")]
    buf = StringIO("".join(lines[data_start:]))
    df = pd.read_csv(buf, sep="\t", header=None, names=cols, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "ID" not in df.columns or "Prediction" not in df.columns:
        raise ValueError(f"SignalP table missing required columns 'ID'/'Prediction'. Got: {list(df.columns)}")
    return df


def _parse_cs_field(cs_field: str) -> Tuple[pd.Series, pd.Series]:
    """
    Parse "CS Position" like: "CS pos: 25-26. Pr: 0.8214"
    Returns (cs_pos, probability) as strings/float-or-NA.
    """
    if not isinstance(cs_field, str) or not cs_field.strip():
        return (pd.NA, pd.NA)
    m = re.search(r"CS pos:\s*([0-9]+-[0-9]+).*?Pr:\s*([0-9.]+)", cs_field)
    if not m:
        return (pd.NA, pd.NA)
    return (m.group(1), float(m.group(2)))


def parse_signalp_txt(txt_path: Path) -> pd.DataFrame:
    """
    Return a DataFrame indexed by sequence ID with columns:
    signalp6_pred (bool), signalp6_type, signalp6_cs, signalp6_prob
    """
    df = _read_signalp_txt(txt_path)

    # Normalize subtype/OTHER to numeric where present
    present_subs = [c for c in SUBTYPE_COLS if c in df.columns]
    for col in present_subs + (["OTHER"] if "OTHER" in df.columns else []):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean: 'SP' == has signal peptide
    pred_is_sp = df["Prediction"].astype(str).str.upper().eq("SP")

    # Type: best-scoring subtype when SP
    sp_type = pd.Series(pd.NA, index=df.index, dtype="object")
    if present_subs:
        sub_df = df[present_subs].astype(float)
        sp_type = sub_df.idxmax(axis=1).where(pred_is_sp, pd.NA)

    # Cleavage site + probability
    cs_vals, prob_vals = zip(*[_parse_cs_field(x) for x in df.get("CS Position", pd.Series([None] * len(df)))])

    out = pd.DataFrame(
        {
            "signalp6_pred": pred_is_sp.fillna(False),
            "signalp6_type": sp_type,
            "signalp6_cs": list(cs_vals),
            "signalp6_prob": list(prob_vals),
        },
        index=df["ID"].astype(str),
    )
    out.index.name = "sequence_id"
    return out


def run_signalp6(
    fasta: Path,
    outdir: Path,
    mode: str = "fast",
    organism: str = "other",
    batch: int = 64,
    threads: int = 1,
    signalp6_bin: str = "signalp6",
) -> Path:
    """Invoke signalp6 and return the path to the result .txt file."""
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        signalp6_bin,
        "--fastafile", str(fasta),
        "--output_dir", str(outdir),
        "--mode", mode,               # fast | slow | slow-sequential
        "--organism", organism,       # other | euk | eukarya
        "--format", "txt",
        "--bsize", str(batch),
    ]
    # Control threads via env (avoid buggy CLI thread flag)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)

    subprocess.run(cmd, check=True, env=env)
    return find_txt_table(outdir)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate PlasticEnz Summary_table.tsv with SignalP 6 predictions."
    )
    p.add_argument("--in", dest="inp", required=True, help="Path to PlasticEnz Summary_table.tsv")
    p.add_argument("--out", dest="outp", required=True, help="Path to write augmented TSV")
    p.add_argument("--threads", type=int, default=1, help="CPU threads for SignalP6")
    p.add_argument("--batch", type=int, default=64, help="SignalP6 batch size")
    p.add_argument("--organism", default="other", choices=["other", "euk", "eukarya"], help="SignalP6 organism model")
    p.add_argument("--mode", default="fast", choices=["fast", "slow", "slow-sequential"], help="SignalP6 mode")
    p.add_argument("--signalp6", default="signalp6", help="Path to signalp6 executable")
    p.add_argument("--tmpdir", default=None, help="Temporary working directory (optional)")
    p.add_argument("--keep-tmp", action="store_true", help="Keep temp workspace with raw SignalP6 outputs")
    args = p.parse_args()

    if shutil.which(args.signalp6) is None and not Path(args.signalp6).exists():
        raise RuntimeError(f"signalp6 not found: {args.signalp6}")

    df = pd.read_csv(args.inp, sep="\t")
    for c in (REQ_ID_COL, REQ_SEQ_COL):
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Found: {list(df.columns)}")

    # temp workspace
    if args.keep_tmp:
        tmpdir = Path(tempfile.mkdtemp(dir=args.tmpdir))
        cleanup = False
    else:
        tmp_ctx = tempfile.TemporaryDirectory(dir=args.tmpdir)
        tmpdir = Path(tmp_ctx.name)
        cleanup = True

    try:
        # FASTA for SignalP
        fasta = tmpdir / "proteins.fa"
        write_fasta(df[[REQ_ID_COL, REQ_SEQ_COL]], fasta)

        # Run SignalP6
        sp_out = tmpdir / "signalp6_out"
        txt_path = run_signalp6(
            fasta,
            sp_out,
            mode=args.mode,
            organism=args.organism,
            batch=args.batch,
            threads=args.threads,
            signalp6_bin=args.signalp6,
        )

        # Parse + merge (derived fields)
        sp_df = parse_signalp_txt(txt_path)
        merged = df.merge(sp_df, how="left", left_on=REQ_ID_COL, right_index=True)

        # Read the raw table with original columns and merge them (excluding 'ID' key)
        raw_df = _read_signalp_txt(txt_path)  # original columns from the txt
        if "ID" not in raw_df.columns:
            raise ValueError("SignalP raw table does not contain 'ID' column needed for merge.")
        raw_df_idx = raw_df.set_index(raw_df["ID"].astype(str))
        raw_cols_to_add = [c for c in raw_df_idx.columns if c != "ID"]  # keep original names
        merged = merged.merge(raw_df_idx[raw_cols_to_add], how="left",
                              left_on=REQ_ID_COL, right_index=True)
        
        # Recompute derived columns from raw fields to avoid empties
        present_subs = [c for c in SUBTYPE_COLS if c in merged.columns]
        
        # 1) signalp6_pred: True if Prediction == "SP", else False
        merged["signalp6_pred"] = merged["Prediction"].astype(str).str.upper().eq("SP").fillna(False)
        
        # 2) signalp6_type: best-scoring subtype when SP, else <NA>
        if present_subs:
            sub_df = merged[present_subs].apply(pd.to_numeric, errors="coerce")
            merged["signalp6_type"] = sub_df.idxmax(axis=1).where(merged["signalp6_pred"], pd.NA)
        else:
            merged["signalp6_type"] = pd.NA

        # 3) signalp6_cs and 4) signalp6_prob from "CS Position"
        cs_vals, prob_vals = zip(*[_parse_cs_field(x) for x in merged.get("CS Position", pd.Series([None]*len(merged)))])
        merged["signalp6_cs"] = list(cs_vals)
        merged["signalp6_prob"] = list(prob_vals)
        merged["signalp6_mode"] = args.mode
        merged["signalp6_organism"] = args.organism
        merged.to_csv(args.outp, sep="\t", index=False)

        # Save raw table for inspection
        raw_copy = Path(args.outp).with_suffix(".signalp6_raw.txt")
        shutil.copy2(txt_path, raw_copy)

        print(f"✅ Wrote: {args.outp}")
        print(f"📝 Raw SignalP table copied to: {raw_copy}")
        print(f"🗂️ Temp workspace: {tmpdir} (keep_tmp={args.keep_tmp})")

    finally:
        if cleanup:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()



