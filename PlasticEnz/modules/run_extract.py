#!/usr/bin/env python3
import os
import re
from Bio import SeqIO

def parse_hmmer_output(hmmer_files):
    """Parse HMMER output files and extract relevant data."""
    hmmer_data = {}
    for file in hmmer_files:
        polymer = os.path.basename(file).split('_')[0]  # Extract polymer name from filename
        with open(file, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                protein_name = parts[0]
                evalue = float(parts[4])
                bitscore = float(parts[5])
                # Store each entry so that later we can group by polymer.
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
        polymer = os.path.basename(file).split('_')[0]  # Extract polymer name from filename
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

def get_protein_codes(protein_file):
    protein_seqs = {}
    for record in SeqIO.parse(protein_file, "fasta"):
        prot_id = record.id.split()[0]  # Use only the first word
        protein_seqs[prot_id] = str(record.seq)
    return protein_seqs

def extract_protein_sequences(protein_file, protein_list, output_file):
    unique_proteins = set(protein_list)
    with open(output_file, "w") as out_fasta:
        for record in SeqIO.parse(protein_file, "fasta"):
            prot_id = record.id.split()[0]  # Normalize the ID
            if prot_id in unique_proteins:
                SeqIO.write(record, out_fasta, "fasta")

def write_tsv_file(protein_data, output_tsv, protein_codes):
    """Write protein data to a TSV file using the extracted protein sequences."""
    with open(output_tsv, "w") as tsv:
        # Changed header "Protein Code" to "Protein Sequence"
        tsv.write("Protein Name\tProtein Sequence\tHomology\tPolymer\tHMMER E-value\tHMMER Bitscore\tDIAMOND E-value\tDIAMOND Bitscore\n")
        for protein_name, entries in protein_data.items():
            unique_polymers = set(entry["polymer"] for entry in entries)
            for poly in unique_polymers:
                hmmer_entry = next((entry for entry in entries if entry["source"] == "HMMER" and entry["polymer"] == poly), {})
                diamond_entry = next((entry for entry in entries if entry["source"] == "DIAMOND" and entry["polymer"] == poly), {})
                hmmer_evalue = hmmer_entry.get("evalue", "NA")
                hmmer_bitscore = hmmer_entry.get("bitscore", "NA")
                diamond_evalue = diamond_entry.get("evalue", "NA")
                diamond_bitscore = diamond_entry.get("bitscore", "NA")
                homology = "+".join(set(entry["source"] for entry in entries if entry["polymer"] == poly))
                protein_seq = protein_codes.get(protein_name, "")
                tsv.write(f"{protein_name}\t{protein_seq}\t{homology}\t{poly}\t{hmmer_evalue}\t{hmmer_bitscore}\t{diamond_evalue}\t{diamond_bitscore}\n")


def run_extract(hmmer_outputs, diamond_outputs, protein_file, output_dir):
    """Export homologues to table and FASTA, and return their paths."""
    summary_file = os.path.join(output_dir, "Summary_table.tsv")
    fasta_file = os.path.join(output_dir, "Proteins_unique.fa")
    
    hmmer_data = parse_hmmer_output(hmmer_outputs)
    diamond_data = parse_diamond_output(diamond_outputs)

    # Merge DIAMOND entries with HMMER entries for each protein.
    protein_data = hmmer_data
    for protein, entries in diamond_data.items():
        if protein not in protein_data:
            protein_data[protein] = entries
        else:
            protein_data[protein].extend(entries)

    unique_proteins = list(protein_data.keys())
    extract_protein_sequences(protein_file, unique_proteins, fasta_file)
    protein_codes = get_protein_codes(protein_file)
    write_tsv_file(protein_data, summary_file, protein_codes)

    # Return paths for downstream use
    return summary_file, fasta_file





