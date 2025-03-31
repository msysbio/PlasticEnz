import os
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from Bio import SeqIO
import platform

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("testing")

def preprocess_sequence(seq):
    seq = seq.replace("X", "").replace("x", "").replace("*", "").upper()
    if not set(seq).issubset(VALID_AMINO_ACIDS):
        print(f"Warning: Sequence '{seq}' contains invalid amino acids. Skipping.")
        return None
    return seq

def load_protbert_model():
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def get_protbert_embedding(sequence, tokenizer, model, device):
    seq_formatted = " ".join(list(sequence))
    encoded_input = tokenizer(seq_formatted, return_tensors="pt")
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy()

def generate_embeddings(fasta_file, tokenizer, model, device, batch_size=64):
    sequence_ids = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        clean_seq = preprocess_sequence(seq)
        if clean_seq is None or len(clean_seq) == 0:
            continue
        sequences.append(clean_seq)
        sequence_ids.append(record.id)
    
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            spaced_sequences = [' '.join(list(seq)) for seq in batch_sequences]
            inputs = tokenizer(spaced_sequences, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            embeddings_batch = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            embeddings.extend(embeddings_batch.cpu().numpy())
    return np.array(embeddings), sequence_ids

def update_summary_table(summary_table, sequence_ids, predictions, col_names=None):
    df = pd.read_csv(summary_table, sep="\t")
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if col_names is None:
        n_cols = predictions.shape[1]
        col_names = [f"Prediction Score {i+1}" for i in range(n_cols)]
    prediction_data = pd.DataFrame(predictions, columns=col_names)
    prediction_data.insert(0, "Protein Name", sequence_ids)
    merged_df = pd.merge(df, prediction_data, on="Protein Name", how="left")
    merged_df.to_csv(summary_table, sep="\t", index=False)

def run_predictions(fasta_file, summary_table, model_path, gpu, polymers=None, model_tag="model"):
    print("🧪Loading ProtBERT model...")
    tokenizer, bert_model, device = load_protbert_model()
    if gpu:
        if torch.cuda.is_available():
            print("✅GPU detected and will be used.")
        else:
            print("❌No GPU detected. Running on CPU.")
            device = torch.device("cpu")
            bert_model.to(device)
    else:
        print("🧪Using CPU for computations.")
        device = torch.device("cpu")
        bert_model.to(device)
    
    print("🧪Generating embeddings using new protocol...")
    embeddings, sequence_ids = generate_embeddings(fasta_file, tokenizer, bert_model, device, batch_size=16)
    
    print("🧪Loading prediction model...")
    ext = os.path.splitext(model_path)[1]
    if ext == ".pt":
        from torch import nn
        class SimpleNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout_prob)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        input_dim = embeddings.shape[1]
        total_labels = 7  # The labels are: PHB, PCL, PET, PLA, PHA, PBSA, PBAT
        hidden_size = 128
        dropout_prob = 0.3
        nn_model = SimpleNN(input_dim, hidden_size, total_labels, dropout_prob).to(device)
        state_dict = torch.load(model_path, map_location=device)
        nn_model.load_state_dict(state_dict)
        nn_model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
            logits = nn_model(input_tensor)
            probs = torch.sigmoid(logits)
        predictions = probs.cpu().numpy()
    else:
        with open(model_path, "rb") as f:
            xgb_model = pickle.load(f)
        preds_list = xgb_model.predict_proba(embeddings)
        predictions = np.hstack([pred[:, 1].reshape(-1, 1) for pred in preds_list])
    
    # --- Filter predictions to output only for target polymers ---
    # Mapping of polymer name to its index (must match training order).
    label_mapping = {
        "PHB": 0,
        "PCL": 1,
        "PET": 2,
        "PLA": 3,
        "PHA": 4,
        "PBSA": 5,
        "PBAT": 6
    }
    # We want predictions only for PET and PHB.
    target_polymers = {"PET", "PHB"}
    prediction_polymers = [poly for poly in polymers if poly in target_polymers] if polymers is not None else list(target_polymers)
    
    if prediction_polymers:
        selected_indices = [label_mapping[poly] for poly in prediction_polymers]
        predictions = predictions[:, selected_indices]
        col_names = [f"{poly}_prediction_{model_tag}" for poly in prediction_polymers]
    else:
        n_cols = predictions.shape[1]
        col_names = [f"Prediction Score {i+1}" for i in range(n_cols)]
    
    # Create a DataFrame from the predictions.
    prediction_data = pd.DataFrame(predictions, columns=col_names)
    prediction_data.insert(0, "Protein Name", sequence_ids)
    
    # Merge the predictions with the summary table.
    df_summary = pd.read_csv(summary_table, sep="\t")
    merged = pd.merge(df_summary, prediction_data, on="Protein Name", how="left")
    
    # For each prediction column, if for a given protein its "Polymers" field does not include the polymer, clear the prediction.
    for poly, col in zip(prediction_polymers, col_names):
        merged.loc[merged["Polymer"].str.upper() != poly, col] = np.nan

    # Optionally, drop any prediction column that is entirely NaN (i.e. not relevant for any protein).
    for col in col_names:
        if merged[col].isna().all():
            merged.drop(columns=[col], inplace=True)
    
    merged.to_csv(summary_table, sep="\t", index=False)
    
    print("✅Predictions added to summary table.")
    print("✅Prediction step completed successfully!")


