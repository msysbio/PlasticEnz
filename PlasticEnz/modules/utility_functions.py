#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:35:46 2025

@author: u0145079
"""

import sys
import pandas as pd
from Bio import SearchIO

VALID_POLYMERS = [
    "LDPE", "PBSA", "PBS", "PCL", "PES", "PHBV", "PLA",
    "P3HP", "PBAT", "PEA", "PET", "PHA", "PHB"
]

def validate_inputs(args):
    """Validate user inputs and ensure correct argument combinations."""
    if args.contigs and args.proteins:
        sys.exit("Error: Provide either --contigs or --proteins, not both.")
    
    if args.genome and args.contigs:
        sys.exit("Error: Provide either --genome or --contigs, not both.")

    if not (args.genome or args.contigs or args.proteins):
        sys.exit("Error: You must provide --genome, --contigs, or --proteins as input.")

    #if args.genome and not (args.reads_forward and args.reads_reverse):
        #sys.exit("Error: When using --genome, you must also provide both forward and reverse reads (-1 and -2).")

def validate_polymers(polymers):
    """Validate that the provided polymers are in the list of valid polymers."""
    if polymers.lower() == "all":
        return VALID_POLYMERS
    polymer_list = [poly.strip().upper() for poly in polymers.split(",")]
    invalid_polymers = [poly for poly in polymer_list if poly not in VALID_POLYMERS]
    if invalid_polymers:
        sys.exit(f"Error: Invalid polymer(s) provided: {', '.join(invalid_polymers)}. Valid options are: {', '.join(VALID_POLYMERS)}")
    return polymer_list

def count_hmmer_hits(hmmer_files):
    """Count the number of hits in HMMER tblout files using Biopython."""
    total_hits = 0
    for hmmer_file in hmmer_files:
        try:
            for result in SearchIO.parse(hmmer_file, "hmmer3-tab"):
                total_hits += len(result.hits)
        except Exception as e:
            print(f"Error reading HMMER file {hmmer_file}: {e}")
            continue
    return total_hits

def count_diamond_hits(diamond_files):
    """Count the number of lines in DIAMOND output files."""
    total_hits = 0
    for diamond_file in diamond_files:
        try:
            with open(diamond_file, "r") as f:
                hits = sum(1 for _ in f)
                total_hits += hits
        except Exception as e:
            print(f"Error reading DIAMOND file {diamond_file}: {e}")
            continue
    return total_hits

def merge_abundance_with_summary(summary_file, abundance_file):
    """
    Merge the abundance data (TPM, RPKM, RAW) with the existing summary table.

    Parameters:
    - summary_file: Path to the existing summary table (TSV).
    - abundance_file: Path to the abundance file (TSV).

    Returns:
    - None (the summary file is updated in place).
    """
    # Load the summary and abundance tables
    summary_df = pd.read_csv(summary_file, sep="\t")
    abundance_df = pd.read_csv(abundance_file, sep="\t")

    # Ensure the two tables have a common column for merging
    if "Protein Name" not in abundance_df.columns:
        raise ValueError("Abundance file must contain a 'Protein Name' column for merging.")

    # Merge on "Protein Name"
    merged_df = pd.merge(summary_df, abundance_df, on="Protein Name", how="left")

    # Save the merged table back to the summary file
    merged_df.to_csv(summary_file, sep="\t", index=False)
    print(f"Merged abundances into {summary_file}.")
