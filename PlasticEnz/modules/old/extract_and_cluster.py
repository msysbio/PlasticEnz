import os
import re
import subprocess
from collections import defaultdict

def extract_prefix_from_fasta(fasta_file):
    basename = os.path.basename(fasta_file)
    if "_proteins" in basename:
        return basename.split("_proteins")[0]
    return basename.split(".")[0]

def extract_polymer_hmmer(filename):
    basename = os.path.basename(filename)
    match = re.search(r"_(.*?)-HMMER", basename)
    if match:
        poly = match.group(1)
        return poly
    return None

def extract_polymer_diamond(filename):
    basename = os.path.basename(filename)
    match = re.search(r"_(.*?)-DIAMOND", basename)
    if match:
        poly = match.group(1)
        return poly
    return None

def read_prodigal_fasta(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as f:
        current_id = None
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = (current_header, "".join(current_seq))
                current_header = line[1:]
                pid = current_header.split()[0]
                current_id = pid
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            sequences[current_id] = (current_header, "".join(current_seq))
    return sequences

def parse_hmmer_tblout(tblout_file, tool_dict):
    polymer = extract_polymer_hmmer(tblout_file)
    # If no polymer identified, skip processing this file
    if polymer is None:
        return
    with open(tblout_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            protein_id = parts[0]
            if protein_id not in tool_dict:
                tool_dict[protein_id] = {"TOOL": set(), "POLYMER": set()}
            tool_dict[protein_id]["TOOL"].add("HMMER")
            tool_dict[protein_id]["POLYMER"].add(polymer)

def parse_diamond_tsv(diamond_file, tool_dict):
    polymer = extract_polymer_diamond(diamond_file)
    if polymer is None:
        return
    with open(diamond_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split('\t')
            protein_id = parts[0]
            if protein_id not in tool_dict:
                tool_dict[protein_id] = {"TOOL": set(), "POLYMER": set()}
            tool_dict[protein_id]["TOOL"].add("DIAMOND")
            tool_dict[protein_id]["POLYMER"].add(polymer)

def write_polymer_fastas(tool_dict, sequences, prefix, output_dir="."):
    """
    Write polymer-specific FASTA files based on the tool_dict and sequences.

    Parameters:
    - tool_dict (dict): Dictionary mapping protein IDs to tools and polymers.
    - sequences (dict): Dictionary mapping protein IDs to (header, sequence).
    - prefix (str): Prefix for output filenames.
    - output_dir (str): Directory to write output files.

    Returns:
    - None
    """
    os.makedirs(output_dir, exist_ok=True)
    polymer_files = {}

    try:
        for protein_id, info in tool_dict.items():
            # Filter out None polymers if any
            polys = [p for p in info["POLYMER"] if p is not None]
            polymers = sorted(polys)
            if not polymers:
                # If no valid polymers, skip
                continue
            for poly in polymers:
                # Construct filename without duplicating 'PROTEINS'
                fasta_name = os.path.join(output_dir, f"{prefix}_PROTEINS_{poly}.fa")
                if poly not in polymer_files:
                    polymer_files[poly] = open(fasta_name, "w")
                if protein_id in sequences:
                    original_header, seq = sequences[protein_id]
                    tools = sorted(info["TOOL"])
                    tool_str = "/".join(tools)
                    poly_str = "/".join(polymers)
                    new_header = f">{protein_id} # {tool_str} # {poly_str}"
                    polymer_files[poly].write(new_header + "\n")
                    polymer_files[poly].write(seq + "\n")
    finally:
        for pf in polymer_files.values():
            pf.close()

def run_cdhit(prefix, output_dir=".", identity=0.95):
    """
    Cluster protein sequences using CD-HIT.

    Parameters:
    - prefix (str): Prefix for input and output filenames.
    - output_dir (str): Directory where output files will be written.
    - identity (float): Sequence identity threshold for clustering.

    Returns:
    - None
    """
    for filename in os.listdir(output_dir):
        if filename.startswith(f"{prefix}_PROTEINS_") and filename.endswith(".fa"):
            # Ensure we are not re-clustering already unique files
            if "_unique.fa" in filename:
                continue  # Skip already unique files
            poly = filename.split("_PROTEINS_")[1].replace(".fa","")
            out_name = os.path.join(output_dir, f"{prefix}_PROTEINS_{poly}_unique.fa")
            input_file = os.path.join(output_dir, filename)
            cmd = ["cd-hit", "-i", input_file, "-o", out_name, "-c", str(identity)]
            subprocess.run(cmd, check=True)

def create_summary(tool_dict, summary_file):
    """
    Create a summary TSV file listing each protein, the tools that detected it, and the polymers it was associated with.

    Parameters:
    - tool_dict (dict): Dictionary mapping protein IDs to tools and polymers.
    - summary_file (str): Path to the summary TSV file.

    Returns:
    - None
    """
    with open(summary_file, "w") as out:
        out.write("Protein_ID\tTOOL\tPolymer\n")
        for protein_id, info in tool_dict.items():
            # Filter None if any
            polys = [p for p in info["POLYMER"] if p is not None]
            if not polys:
                continue
            tools = "/".join(sorted(info["TOOL"]))
            polys_str = "/".join(sorted(polys))
            out.write(f"{protein_id}\t{tools}\t{polys_str}\n")

def process_files(prodigal_fasta, output_prefix, input_dir=".", output_dir=".", cdhit_identity=0.95):
    """
    Process HMMER and DIAMOND output files, cluster sequences using CD-HIT, and create a summary.

    Parameters:
    - prodigal_fasta (str): Filename of the Prodigal-produced protein FASTA.
    - output_prefix (str): Prefix for output files.
    - input_dir (str): Directory containing input files.
    - output_dir (str): Directory to write output files.
    - cdhit_identity (float): Identity threshold for CD-HIT clustering.

    Returns:
    - summary_file (str): Path to the summary TSV file.
    """
    sequences = read_prodigal_fasta(os.path.join(input_dir, prodigal_fasta))
    tool_dict = {}

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if filename.endswith(".tblout") and "-HMMER" in filename:
            parse_hmmer_tblout(filepath, tool_dict)
        elif filename.endswith(".tsv") and "-DIAMOND" in filename:
            parse_diamond_tsv(filepath, tool_dict)

    write_polymer_fastas(tool_dict, sequences, output_prefix, output_dir=output_dir)
    run_cdhit(output_prefix, output_dir=output_dir, identity=cdhit_identity)

    summary_file = os.path.join(output_dir, f"{output_prefix}_protein_summary.tsv")
    create_summary(tool_dict, summary_file)
    return summary_file




