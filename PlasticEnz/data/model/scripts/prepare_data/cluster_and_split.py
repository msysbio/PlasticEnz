import os
import random
from Bio import SeqIO
import pandas as pd
import subprocess

def label_sequences(input_fasta, label):
    """
    Add a label to sequence headers.
    """
    labeled_sequences = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        record.id = f"{record.id}_{label}"
        record.description = ""
        labeled_sequences.append(record)
    return labeled_sequences

def save_fasta(sequences, output_fasta):
    """
    Save sequences to a FASTA file.
    """
    SeqIO.write(sequences, output_fasta, "fasta")
    print(f"FASTA file saved to {output_fasta}.")

def run_cd_hit(input_fasta, output_fasta, similarity=0.95):
    """
    Run CD-Hit clustering.
    """
    print(f"Running CD-Hit at {similarity*100}% similarity...")
    command = [
        "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/External_tools/cd-hit-v4.8.1-2019-0228/cd-hit",  # Replace with the path to your CD-Hit executable
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(similarity),
        "-d", "0",  # Full header
        "-n", "5",  # Word size
        "-M", "16000",  # Memory limit in MB
        "-T", "4"  # Number of threads
    ]
    subprocess.run(command, check=True)
    print(f"CD-Hit clustering completed. Output saved to {output_fasta}.")

def parse_cd_hit_clusters(clstr_file):
    """
    Parse CD-Hit cluster output to group sequences into clusters.
    """
    clusters = {}
    current_cluster = None
    with open(clstr_file) as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = line.strip()
                clusters[current_cluster] = []
            else:
                seq_id = line.split(">")[1].split("...")[0]
                clusters[current_cluster].append(seq_id)
    return clusters

def split_clusters(clusters, test_size=0.2):
    """
    Split clusters into training and testing sets.
    """
    cluster_ids = list(clusters.keys())
    random.shuffle(cluster_ids)

    split_index = int(len(cluster_ids) * (1 - test_size))
    train_clusters = cluster_ids[:split_index]
    test_clusters = cluster_ids[split_index:]

    train_ids = {seq_id for cluster in train_clusters for seq_id in clusters[cluster]}
    test_ids = {seq_id for cluster in test_clusters for seq_id in clusters[cluster]}

    return train_ids, test_ids

def write_tsv(sequences, output_tsv):
    """
    Write sequences to a TSV file.
    """
    data = []
    for record in sequences:
        label = 1 if "positive" in record.id else 0
        data.append([record.id, str(record.seq), label])
    
    df = pd.DataFrame(data, columns=["id", "sequence", "label"])
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"TSV file saved to {output_tsv}.")

def main():
    # Input files
    positive_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/positive_set.fa"
    negative_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/negative_set.fasta"
    combined_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/combined.fasta"
    clustered_fasta = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/clustered.fasta"
    clustered_clstr = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/clustered.fasta.clstr"

    # Output TSV files
    train_tsv = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/train.tsv"
    test_tsv = "/Users/u0145079/Library/CloudStorage/OneDrive-KULeuven/Desktop/Doctorate/PlasticEnz_toolset/Model_making/test.tsv"

    # Label sequences
    print("Labeling sequences...")
    positive_sequences = label_sequences(positive_fasta, "positive")
    negative_sequences = label_sequences(negative_fasta, "negative")

    # Combine into one FASTA
    combined_sequences = positive_sequences + negative_sequences
    save_fasta(combined_sequences, combined_fasta)

    # Run CD-Hit clustering
    run_cd_hit(combined_fasta, clustered_fasta)

    # Parse clusters
    print("Parsing CD-Hit clusters...")
    clusters = parse_cd_hit_clusters(clustered_clstr)

    # Split clusters into training and testing sets
    print("Splitting clusters into training and testing sets...")
    train_ids, test_ids = split_clusters(clusters)

    # Separate sequences into training and testing sets
    train_sequences = []
    test_sequences = []

    for record in SeqIO.parse(clustered_fasta, "fasta"):
        if record.id in train_ids:
            train_sequences.append(record)
        elif record.id in test_ids:
            test_sequences.append(record)

    # Write training and testing TSV files
    print("Writing TSV files...")
    write_tsv(train_sequences, train_tsv)
    write_tsv(test_sequences, test_tsv)

if __name__ == "__main__":
    main()
