#!/usr/bin/env python3
"""
Prediction script for protein binding site predictor.

This script loads a trained model and makes predictions on protein sequences.
Supports both single protein prediction and batch processing.
"""

import argparse
import os
import sys
import yaml
import pickle
import timeit
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.utils.data.sampler as sampler
import numpy as np
import pandas as pd
from torch.utils import data

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model import (
    Predictor, Encoder, Decoder, DecoderLayer, 
    SelfAttention, PositionwiseFeedforward, Tester
)
from src.utils.metrics import metrics


class PredictionDataSet(data.Dataset):
    """Dataset class for prediction on new protein sequences."""
    
    def __init__(self, window_size: int, encode_data: List, label_data: List, protein_list: List):
        super(PredictionDataSet, self).__init__()
        self.all_encodes = encode_data
        self.all_label = label_data
        self.protein_list = protein_list
        self.window_size = window_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get item from dataset."""
        ii, protein_id, seq_length = self.protein_list[index]
        window_size = self.window_size

        win_start = ii - window_size
        win_end = ii + window_size
        seq_length = int(seq_length)
        label_idx = (win_start + win_end) // 2

        all_seq_features = self.all_encodes

        # Create local features with padding
        local_features = []
        while win_start < 0:
            acid_embed = [0 for _ in range(len(self.all_encodes[0]))]
            local_features.append(acid_embed)
            win_start += 1

        valid_end = min(win_end, seq_length - 1)
        while win_start <= valid_end:
            acid_embed = self.all_encodes[win_start]
            local_features.append(acid_embed)
            win_start += 1

        while win_start <= win_end:
            acid_embed = [0 for _ in range(len(self.all_encodes[0]))]
            local_features.append(acid_embed)
            win_start += 1

        label = self.all_label[label_idx]
        label = np.array(label, dtype=np.float32)
        all_seq_features = np.stack(all_seq_features).astype(float)
        local_features = np.stack(local_features).astype(float)
        
        return local_features, all_seq_features, label

    def __len__(self) -> int:
        return len(self.protein_list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict protein binding sites")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model file")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input protein data directory or file")
    parser.add_argument("--output", type=str, required=True, 
                       help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=0.3, 
                       help="Prediction threshold for binary classification")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'auto', 'cpu', 'cuda', or 'cuda:X'")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    
    return device


def create_model(config: Dict, device: torch.device) -> Predictor:
    """Create model architecture."""
    protein_dim = config['model']['protein_dim']
    local_dim = config['model']['local_dim']
    hid_dim = config['model']['hidden_dim']
    n_layers = config['model']['n_layers']
    n_heads = config['model']['n_heads']
    pf_dim = config['model']['pf_dim']
    dropout = config['model']['dropout']
    kernel_size = config['model']['kernel_size']
    
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, 
                     SelfAttention, PositionwiseFeedforward, dropout, device)
    
    model = Predictor(encoder, decoder, device)
    return model


def load_model(model_path: str, model: Predictor, device: torch.device) -> Predictor:
    """Load trained model weights."""
    print(f"Loading model from {model_path}")
    
    if device.type == 'cpu':
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model


def load_protein_data(input_path: str) -> Tuple[List, List, pd.DataFrame]:
    """Load protein data for prediction."""
    print(f"Loading protein data from {input_path}")
    
    # Determine if input is directory or file
    input_path = Path(input_path)
    
    if input_path.is_dir():
        # Directory mode - load cached embeddings
        encode_file = input_path / "encoded_data.pkl"
        label_file = input_path / "labels.pkl"
        
        with open(encode_file, "rb") as f:
            all_encodes = pickle.load(f)
        
        with open(label_file, "rb") as f:
            all_labels = pickle.load(f)
        
        # Create dummy dataframe for compatibility
        protein_df = pd.DataFrame({
            'name': [f'protein_{i}' for i in range(len(all_encodes))],
            'sequence': [''] * len(all_encodes),
            'label': [''] * len(all_labels)
        })
        
    else:
        # File mode - parse FASTA-like format
        all_encodes = []
        all_labels = []
        
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        data_lines = [line.strip() for line in lines]
        
        # Parse protein data (assuming format: >name, sequence, labels)
        names, sequences, labels = [], [], []
        i = 0
        while i < len(data_lines):
            if data_lines[i].startswith('>'):
                name = data_lines[i][1:]  # Remove '>'
                sequence = data_lines[i + 1] if i + 1 < len(data_lines) else ''
                label = data_lines[i + 2] if i + 2 < len(data_lines) else ''
                
                names.append(name)
                sequences.append(sequence)
                labels.append(label)
                i += 3
            else:
                i += 1
        
        protein_df = pd.DataFrame({
            'name': names,
            'sequence': sequences,
            'label': labels
        })
        
        # Load corresponding embeddings (this would need to be pre-computed)
        # For now, assume embeddings are provided separately
        encode_file = input_path.parent / f"{input_path.stem}_embeddings.pkl"
        label_file = input_path.parent / f"{input_path.stem}_labels.pkl"
        
        if encode_file.exists() and label_file.exists():
            with open(encode_file, "rb") as f:
                all_encodes = pickle.load(f)
            with open(label_file, "rb") as f:
                all_labels = pickle.load(f)
        else:
            raise FileNotFoundError(f"Embedding files not found: {encode_file}, {label_file}")
    
    print(f"Loaded {len(all_encodes)} proteins for prediction")
    return all_encodes, all_labels, protein_df


def stack_fn(batch: List) -> Tuple:
    """Collate function for DataLoader."""
    local_features, all_seq_features, labels = [], [], []
    
    for local, seq, label in batch:
        local_features.append(local)
        all_seq_features.append(seq)
        labels.append(label)
    
    # Find maximum lengths
    locals_len = max(local.shape[0] for local in local_features)
    proteins_len = max(protein.shape[0] for protein in all_seq_features)
    
    N = len(local_features)
    local_num = [local.shape[0] for local in local_features]
    protein_num = [protein.shape[0] for protein in all_seq_features]

    local_dim = 2560
    protein_dim = 2560
    
    # Pad sequences
    locals_new = np.zeros((N, locals_len, local_dim))
    for i, local in enumerate(local_features):
        a_len = local.shape[0]
        locals_new[i, :a_len, :] = local

    proteins_new = np.zeros((N, proteins_len, protein_dim))
    for i, protein in enumerate(all_seq_features):
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
        
    labels_new = np.zeros(N, dtype=np.long)
    for i, label in enumerate(labels):
        labels_new[i] = label

    return (np.stack([locals_new]), np.stack([proteins_new]), 
            np.stack([labels_new]), local_num, protein_num)


def predict_protein(model: Predictor, tester: Tester, protein_data: Tuple, 
                   device: torch.device, batch_size: int, threshold: float) -> Tuple:
    """Make predictions for a single protein."""
    all_encodes, all_labels, protein_df = protein_data
    
    predictions_all = []
    
    for index in range(len(all_encodes)):
        print(f"Predicting protein {index + 1}/{len(all_encodes)}")
        
        protein = all_encodes[index]
        label = all_labels[index]
        
        # Create protein list for dataset
        name = protein_df['name'].iloc[index]
        seq = protein_df['sequence'].iloc[index]
        length = len(protein)
        
        protein_list = []
        for i in range(len(seq) if seq else length):
            protein_list.append((i, name, length))
        
        # Create dataset and dataloader
        window_size = 0
        dataset = PredictionDataSet(window_size, protein, label, protein_list)
        
        test_sampler = sampler.SequentialSampler(protein_list)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler,
            num_workers=0, collate_fn=stack_fn, drop_last=False
        )
        
        # Make predictions
        _, predicted_labels, predicted_scores = tester.test(dataloader, device)
        
        # Apply threshold
        predicted_binary = [1 if score >= threshold else 0 for score in predicted_scores]
        
        predictions_all.append({
            'name': name,
            'sequence': seq,
            'predicted_scores': predicted_scores,
            'predicted_labels': predicted_binary,
            'true_labels': list(map(int, protein_df['label'].iloc[index])) if protein_df['label'].iloc[index] else []
        })
    
    return predictions_all


def save_predictions(predictions: List[Dict], output_path: str, threshold: float):
    """Save predictions to files."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions file
    pred_file = output_path / f"predictions_threshold_{threshold:.2f}.txt"
    metrics_file = output_path / f"metrics_threshold_{threshold:.2f}.txt"
    
    with open(pred_file, 'w') as f, open(metrics_file, 'w') as m:
        # Write headers
        m.write('name\tlength\tACC\tAUC\tRec\tPre\tF1\tMCC\tPRC\n')
        
        all_true_labels = []
        all_pred_labels = []
        all_pred_scores = []
        
        for pred in predictions:
            name = pred['name']
            sequence = pred['sequence']
            pred_labels = pred['predicted_labels']
            pred_scores = pred['predicted_scores']
            true_labels = pred['true_labels']
            
            # Write prediction file
            f.write(f'>{name}\n')
            f.write(f'{sequence}\n')
            f.write(''.join(map(str, pred_labels)) + '\n')
            
            # Calculate metrics if true labels available
            if true_labels:
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)
                all_pred_scores.extend(pred_scores)
                
                ACC, AUC, Rec, Pre, F1, MCC, PRC = metrics(
                    true_labels, pred_labels, pred_scores
                )
                
                m.write(f'{name}\t{len(sequence)}\t{ACC:.4f}\t{AUC:.4f}\t'
                       f'{Rec:.4f}\t{Pre:.4f}\t{F1:.4f}\t{MCC:.4f}\t{PRC:.4f}\n')
        
        # Calculate overall metrics
        if all_true_labels:
            ACC_all, AUC_all, Rec_all, Pre_all, F1_all, MCC_all, PRC_all = metrics(
                all_true_labels, all_pred_labels, all_pred_scores
            )
            
            print(f"\nOverall Metrics (threshold={threshold}):")
            print(f"ACC: {ACC_all:.4f}")
            print(f"AUC: {AUC_all:.4f}")
            print(f"Recall: {Rec_all:.4f}")
            print(f"Precision: {Pre_all:.4f}")
            print(f"F1: {F1_all:.4f}")
            print(f"MCC: {MCC_all:.4f}")
            print(f"PRC: {PRC_all:.4f}")
            
            m.write(f'OVERALL\t-\t{ACC_all:.4f}\t{AUC_all:.4f}\t'
                   f'{Rec_all:.4f}\t{Pre_all:.4f}\t{F1_all:.4f}\t{MCC_all:.4f}\t{PRC_all:.4f}\n')
    
    print(f"\nPredictions saved to: {pred_file}")
    print(f"Metrics saved to: {metrics_file}")


def main():
    """Main prediction function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create and load model
    model = create_model(config, device)
    model = load_model(args.model, model, device)
    
    # Create tester
    tester = Tester(model)
    
    # Load protein data
    protein_data = load_protein_data(args.input)
    
    # Make predictions
    print("Starting prediction...")
    start_time = timeit.default_timer()
    
    predictions = predict_protein(
        model, tester, protein_data, device, args.batch_size, args.threshold
    )
    
    end_time = timeit.default_timer()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")
    
    # Save results
    save_predictions(predictions, args.output, args.threshold)
    
    print("Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()