#!/usr/bin/env python3

import os
from Bio import SeqIO

def parse_hmmer_output(hmmer_files):
    """Parse HMMER output files and extract relevant data."""
    hmmer_data = {}
    for file in hmmer_files:
        polymer = os.path.basename(file).split('_')[1]  # Extract polymer name
        with open(file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                protein_name = parts[0]
                evalue = float(parts[4])
                bitscore = float(parts[5])
                hmmer_data.setdefault(protein_name, []).append({
                    "source": "HMMER",
                    "polymer": polymer,
                    "evalue": evalue,
                    "bitscore": bitscore
                })
    return hmmer_data

def parse_diamond_output(diamond_files):
    """Parse DIAMOND output files and extract relevant data."""
    diamond_data = {}
    for file in diamond_files:
        polymer = os.path.basename(file).split('_')[1]  # Extract polymer name
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                protein_name = parts[0]
                evalue = float(parts[10])
                bitscore = float(parts[11])
                diamond_data.setdefault(protein_name, []).append({
                    "source": "DIAMOND",
                    "polymer": polymer,
                    "evalue": evalue,
                    "bitscore": bitscore
                })
    return diamond_data

def extract_protein_sequences(protein_file, protein_list, output_file):
    """Extract unique protein sequences and write them to a FASTA file."""
    unique_proteins = set(protein_list)
    with open(output_file, "w") as out_fasta:
        for record in SeqIO.parse(protein_file, "fasta"):
            if record.id in unique_proteins:
                SeqIO.write(record, out_fasta, "fasta")

def write_tsv_file(protein_data, output_tsv):
    """Write protein data to a TSV file."""
    with open(output_tsv, "w") as tsv:
        tsv.write("Protein Name\tProtein Code\tHomology\tHMMER E-value\tHMMER Bitscore\tDIAMOND E-value\tDIAMOND Bitscore\tPolymers\n")
        for protein_name, entries in protein_data.items():
            hmmer_entry = next((entry for entry in entries if entry["source"] == "HMMER"), {})
            diamond_entry = next((entry for entry in entries if entry["source"] == "DIAMOND"), {})
            
            hmmer_evalue = hmmer_entry.get("evalue", "NA")
            hmmer_bitscore = hmmer_entry.get("bitscore", "NA")
            diamond_evalue = diamond_entry.get("evalue", "NA")
            diamond_bitscore = diamond_entry.get("bitscore", "NA")
            polymers = ",".join(set(entry["polymer"] for entry in entries))
            homology = "+".join(set(entry["source"] for entry in entries))
            
            tsv.write(f"{protein_name}\t{protein_name}\t{homology}\t{hmmer_evalue}\t{hmmer_bitscore}\t{diamond_evalue}\t{diamond_bitscore}\t{polymers}\n")

def run_extract(hmmer_outputs, diamond_outputs, protein_file, output_dir):
    """Export homologues to table and fasta, and return their paths."""
    base_name = os.path.basename(protein_file).split('.')[0]
    output_tsv = os.path.join(output_dir, f"{base_name}_summary.tsv")
    output_fasta = os.path.join(output_dir, f"{base_name}_unique.fasta")
    
    hmmer_data = parse_hmmer_output(hmmer_outputs)
    diamond_data = parse_diamond_output(diamond_outputs)

    protein_data = hmmer_data
    for protein, entries in diamond_data.items():
        if protein not in protein_data:
            protein_data[protein] = entries
        else:
            protein_data[protein].extend(entries)

    unique_proteins = list(protein_data.keys())
    extract_protein_sequences(protein_file, unique_proteins, output_fasta)
    write_tsv_file(protein_data, output_tsv)

    # Return paths for downstream use
    return output_tsv, output_fasta


