#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:40:15 2025

@author: u0145079
"""

import os
import pickle
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

def generate_embeddings(fasta_file, tokenizer, model, device, batch_size=64):
    """Generate ProtBERT embeddings for protein sequences."""
    sequences = []
    sequence_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_ids.append(record.id)
        sequences.append(str(record.seq))

    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            batch_sequences = [seq[:1022] for seq in batch_sequences]  # Truncate sequences longer than 1022 amino acids
            spaced_sequences = [' '.join(list(seq)) for seq in batch_sequences]

            inputs = tokenizer(spaced_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            embeddings_batch = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            embeddings.extend(embeddings_batch.cpu().numpy())

    return np.array(embeddings), sequence_ids

def predict_plastic_degrading_proteins(embeddings, model_path):
    """Predict probabilities for proteins being plastic-degrading using a random forest model."""
    with open(model_path, "rb") as f:
        rf_model = pickle.load(f)

    probabilities = rf_model.predict_proba(embeddings)[:, 1]  # Get the probability of the positive class
    return probabilities


def update_summary_table(summary_table, sequence_ids, probabilities):
    """Update the summary table with prediction scores only."""
    df = pd.read_csv(summary_table, sep="\t")
    prediction_data = pd.DataFrame({
        "Protein Name": sequence_ids,
        "Prediction Score": probabilities
    })
    merged_df = pd.merge(df, prediction_data, on="Protein Name", how="left")
    merged_df.to_csv(summary_table, sep="\t", index=False)

def run_predictions(fasta_file, summary_table, model_path, gpu):
    """Run the prediction pipeline."""
    # Load ProtBERT
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert")

    if gpu:  # Check if GPU is requested
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU detected and will be used.")
        else:
            print("No GPU detected. Please install CUDA for GPU support.")
            device = torch.device("cpu")
    else:
        print("Using CPU for computations.")
        device = torch.device("cpu")

    model.to(device)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings, sequence_ids = generate_embeddings(fasta_file, tokenizer, model, device)

    # Predict using random forest
    print("Making predictions...")
    probabilities = predict_plastic_degrading_proteins(embeddings, model_path)

    # Update summary table
    print("Updating summary table...")
    update_summary_table(summary_table, sequence_ids, probabilities)
    print("Predictions added to summary table.")
    print("Prediction step completed successfully!")


