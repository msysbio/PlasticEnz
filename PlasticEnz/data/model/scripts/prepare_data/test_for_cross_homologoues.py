#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:37:47 2025

@author: u0145079
"""

import os
import pandas as pd
import subprocess
from Bio import SeqIO

def tsv_to_fasta(tsv_file, fasta_file):
    """
    Convert a TSV file with columns [id, sequence, label] to a FASTA file.
    """
    print(f"Converting {tsv_file} to {fasta_file}...")
    df = pd.read_csv(tsv_file, sep="\t")
    with open(fasta_file, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")
    print(f"FASTA file saved: {fasta_file}")

def run_cd_hit_2d(train_fasta, test_fasta, output_file, similarity=0.95):
    """
    Run CD-Hit-2D to check for homologous proteins between training and testing sets.
    """
    print(f"Running CD-Hit-2D at {similarity * 100}% similarity...")
    command = [
        "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/External_tools/cd-hit-v4.8.1-2019-0228/cd-hit-2d",  # Replace with the path to your CD-Hit-2D executable
        "-i", train_fasta,    # Training set
        "-i2", test_fasta,    # Testing set
        "-o", output_file,    # Output file
        "-c", str(similarity),  # Similarity threshold
        "-n", "5",            # Word size
        "-M", "16000",        # Memory limit in MB
        "-T", "4"             # Number of threads
    ]
    subprocess.run(command, check=True)
    print(f"CD-Hit-2D completed. Output saved: {output_file}")

def check_homology(output_file):
    """
    Parse the CD-Hit-2D output to check for homologous sequences.
    """
    homologous_found = False
    with open(output_file, "r") as f:
        for line in f:
            if line.startswith(">") and "similarity" in line:
                homologous_found = True
                break
    if homologous_found:
        print("Homologous sequences found between training and testing sets!")
    else:
        print("No homologous sequences found between training and testing sets.")

def main():
    # Paths to input TSV files
    train_tsv = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/train.tsv"
    test_tsv = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/test.tsv"

    # Temporary FASTA files
    train_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/train-tmp.fasta"
    test_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/test-tmp.fasta"
    cd_hit_output = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/cd_hit_2d_output"

    # Convert TSV files to FASTA format
    tsv_to_fasta(train_tsv, train_fasta)
    tsv_to_fasta(test_tsv, test_fasta)

    # Run CD-Hit-2D
    run_cd_hit_2d(train_fasta, test_fasta, cd_hit_output)

    # Check for homologous sequences
    check_homology(cd_hit_output + ".clstr")  # CD-Hit-2D output cluster file

if __name__ == "__main__":
    main()
