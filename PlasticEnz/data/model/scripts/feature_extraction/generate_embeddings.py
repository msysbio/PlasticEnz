#!/usr/bin/env python3

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data(csv_path):
    """Load sequences and labels from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if 'sequence' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV file must contain 'sequence' and 'label' columns.")
        sequences = df['sequence'].astype(str).tolist()
        labels = df['label'].tolist()
        logging.info(f"Loaded {len(sequences)} sequences from {csv_path}.")
        return sequences, labels
    except Exception as e:
        logging.error(f"Error loading data from {csv_path}: {e}")
        raise e

def validate_sequences(sequences):
    """Validate sequences to ensure they contain only standard 20 amino acids."""
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_indices = []
    for idx, seq in enumerate(sequences):
        if not set(seq.upper()).issubset(valid_amino_acids):
            invalid_indices.append(idx)
            logging.warning(f"Sequence at index {idx} contains invalid amino acids: {seq}")
    if invalid_indices:
        logging.info(f"Found {len(invalid_indices)} invalid sequences. They will be excluded from embedding.")
    return invalid_indices

def embed_sequences_protbert(sequences, tokenizer, model, device, batch_size=64):
    """Generate embeddings for a list of protein sequences in batches."""
    embeddings = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    model.eval()
    with torch.no_grad():
        for batch_num, i in enumerate(tqdm(range(0, len(sequences), batch_size),
                                           desc="Embedding Batches",
                                           total=total_batches), start=1):
            try:
                batch_sequences = sequences[i:i+batch_size]
                # Truncate sequences longer than 1022 amino acids
                batch_sequences = [seq[:1022] for seq in batch_sequences]
                # Add space between amino acids as required by ProtBERT tokenizer
                spaced_sequences = [' '.join(list(seq)) for seq in batch_sequences]
                
                # Use mixed precision if CUDA is available
                if device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        inputs = tokenizer(spaced_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                else:
                    inputs = tokenizer(spaced_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                
                attention_mask = inputs['attention_mask']
                # Compute mean pooling
                embeddings_batch = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                embeddings.extend(embeddings_batch.cpu().numpy())
            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {e}")
                print(f"Error processing batch {batch_num}: {e}")
    
    return np.array(embeddings)

def main():
    # Define paths
    data_dir = '/staging/leuven/stg_00106/PlasticDaphnia-PlasticTool/model'        # **Update this path**
    embeddings_dir = '/staging/leuven/stg_00106/PlasticDaphnia-PlasticTool/model/embeddings'      # **Update this path**
    os.makedirs(embeddings_dir, exist_ok=True)
    
    train_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'test.csv')
    
    # Load data
    logging.info("Loading training data...")
    train_sequences, train_labels = load_data(train_csv)
    logging.info(f"Training sequences: {len(train_sequences)}")
    
    logging.info("Loading testing data...")
    test_sequences, test_labels = load_data(test_csv)
    logging.info(f"Testing sequences: {len(test_sequences)}")
    
    # Validate sequences
    logging.info("Validating sequences...")
    invalid_train = validate_sequences(train_sequences)
    invalid_test = validate_sequences(test_sequences)
    
    if invalid_train:
        train_sequences = [seq for idx, seq in enumerate(train_sequences) if idx not in invalid_train]
        train_labels = [label for idx, label in enumerate(train_labels) if idx not in invalid_train]
        logging.info(f"Removed {len(invalid_train)} invalid training sequences.")
    
    if invalid_test:
        test_sequences = [seq for idx, seq in enumerate(test_sequences) if idx not in invalid_test]
        test_labels = [label for idx, label in enumerate(test_labels) if idx not in invalid_test]
        logging.info(f"Removed {len(invalid_test)} invalid testing sequences.")
    
    # Initialize ProtBERT
    logging.info("Loading ProtBERT tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = AutoModel.from_pretrained("Rostlab/prot_bert")
    except Exception as e:
        logging.error(f"Error loading ProtBERT tokenizer/model: {e}")
        sys.exit(1)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logging.info("CUDA is available. Utilizing GPU for embedding generation.")
    else:
        logging.info("CUDA is not available. Utilizing CPU for embedding generation.")
        torch.set_num_threads(36)  # Optimize for 36 CPU cores
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Generate embeddings for training data
    logging.info("Generating embeddings for training data...")
    train_embeddings = embed_sequences_protbert(train_sequences, tokenizer, model, device, batch_size=64)
    
    # Generate embeddings for testing data
    logging.info("Generating embeddings for testing data...")
    test_embeddings = embed_sequences_protbert(test_sequences, tokenizer, model, device, batch_size=64)
    
    # Save embeddings and labels
    logging.info("Saving embeddings and labels...")
    np.save(os.path.join(embeddings_dir, 'train_embeddings.npy'), train_embeddings)
    np.save(os.path.join(embeddings_dir, 'test_embeddings.npy'), test_embeddings)
    np.save(os.path.join(embeddings_dir, 'train_labels.npy'), np.array(train_labels))
    np.save(os.path.join(embeddings_dir, 'test_labels.npy'), np.array(test_labels))
    logging.info("Embeddings and labels saved successfully.")

if __name__ == "__main__":
    main()

