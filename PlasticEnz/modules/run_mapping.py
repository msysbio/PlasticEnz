#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:10:24 2024

@author: u0145079
"""

import os
import subprocess
import pandas as pd

def run_bowtie2(build_dir, reads_forward, reads_reverse, genes_file, outdir, sample_name):
    """
    Map reads to genes using Bowtie2 and generate a BAM file for a single sample.

    Parameters:
    - build_dir: Directory to store Bowtie2 index files.
    - reads_forward: Path to the forward reads file (FASTQ).
    - reads_reverse: Path to the reverse reads file (FASTQ).
    - genes_file: Path to the genes file (FASTA).
    - outdir: Directory to store the output BAM file.
    - sample_name: Name of the sample (used in output filenames).

    Returns:
    - Path to the sorted BAM file for the sample.
    """
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # Build Bowtie2 index (only needs to be done once)
    bowtie_index = os.path.join(build_dir, "genes_index")
    if not os.path.exists(f"{bowtie_index}.1.bt2"):
        with open(os.path.join(outdir, "bowtie2_build.log"), "w") as log:
            subprocess.check_call(["bowtie2-build", genes_file, bowtie_index], stdout=log, stderr=log)

    # Map reads to genes
    sam_file = os.path.join(outdir, f"{sample_name}_aligned.sam")
    bam_file = os.path.join(outdir, f"{sample_name}_aligned.bam")
    log_file = os.path.join(outdir, f"{sample_name}_bowtie2.log")
    
    # Run Bowtie2 and redirect all output to the log file
    cmd = [
        "bowtie2",
        "-x", bowtie_index,
        "-1", reads_forward,
        "-2", reads_reverse,
        "-S", sam_file,
    ]
    with open(log_file, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)

    # Sort SAM into BAM using samtools
    cmd_sort = ["samtools", "sort", "-o", bam_file, sam_file]
    subprocess.check_call(cmd_sort)

    # Index BAM file
    subprocess.check_call(["samtools", "index", bam_file])

    # Clean up intermediate SAM file
    os.remove(sam_file)

    return bam_file



def calculate_abundance(bam_file, genes_file):
    idxstats_output = subprocess.check_output(["samtools", "idxstats", bam_file]).decode("utf-8")
    gene_lengths = {}
    for line in idxstats_output.strip().split("\n"):
        fields = line.split("\t")
        gene_id = fields[0]
        length = int(fields[1])
        raw_count = int(fields[2])
        if raw_count > 0:
            gene_lengths[gene_id] = {"length": length, "raw_count": raw_count}

    total_mapped_reads = sum([data["raw_count"] for data in gene_lengths.values()])
    rpkm_factors = []
    for gene_id, data in gene_lengths.items():
        rpk = data["raw_count"] / (data["length"] / 1000)
        rpkm = rpk / (total_mapped_reads / 1e6)
        gene_lengths[gene_id]["RPKM"] = rpkm
        rpkm_factors.append(rpk)

    rpkm_sum = sum(rpkm_factors)
    for gene_id, data in gene_lengths.items():
        tpm = (data["raw_count"] / (data["length"] / 1000)) / rpkm_sum * 1e6
        gene_lengths[gene_id]["TPM"] = tpm

    df = pd.DataFrame.from_dict(gene_lengths, orient="index").reset_index()
    df.rename(columns={"index": "Protein Name"}, inplace=True)

    return df


def process_multiple_samples(forward_reads, reverse_reads, genes_file, outdir):
    if len(forward_reads) != len(reverse_reads):
        raise ValueError("Mismatched number of forward and reverse read files.")

    combined_df = pd.DataFrame()

    for forward, reverse in zip(forward_reads, reverse_reads):
        sample_name = os.path.basename(forward).split(".")[0]

        # Run Bowtie2 for the sample
        bam_file = run_bowtie2(
            build_dir=os.path.join(outdir, "bowtie_index"),
            reads_forward=forward,
            reads_reverse=reverse,
            genes_file=genes_file,
            outdir=os.path.join(outdir, sample_name),
            sample_name=sample_name,
        )

        # Calculate abundance
        abundance_df = calculate_abundance(bam_file, genes_file)
        abundance_df["Sample"] = sample_name
        combined_df = pd.concat([combined_df, abundance_df], ignore_index=True)

    return combined_df


def run_mapping(forward_reads, reverse_reads, genes_file, outdir):
    combined_df = process_multiple_samples(forward_reads, reverse_reads, genes_file, outdir)

    # Save combined DataFrame to final output
    combined_output = os.path.join(outdir, "combined_abundance_table.tsv")
    combined_df.to_csv(combined_output, sep="\t", index=False)

    return combined_output


