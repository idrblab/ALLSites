"""
Simplified and robust data generator for protein binding site prediction.
"""

import pickle
import numpy as np
import torch
from typing import List, Tuple
from torch.utils import data


class dataSet(data.Dataset):
    """Simplified dataset class for protein binding site prediction."""
    
    def __init__(self, window_size: int, encode_file: str, label_file: str, list_file: str):
        super(dataSet, self).__init__()
        
        print(f"Loading data from: {encode_file}")
        
        # Load data files
        try:
            with open(encode_file, "rb") as fp:
                self.all_encodes = pickle.load(fp)
            print(f"Loaded {len(self.all_encodes)} protein encodings")
            
            with open(label_file, "rb") as fp:
                self.all_label = pickle.load(fp)
            print(f"Loaded {len(self.all_label)} protein labels")
            
            with open(list_file, "rb") as fp:
                protein_index = pickle.load(fp)
            print(f"Loaded {len(protein_index)} protein indices")
            
        except Exception as e:
            print(f"Error loading data files: {e}")
            raise
        
        # Process protein list
        self.protein_list = []
        for item in protein_index:
            try:
                if len(item) >= 6:
                    count, id_idx, ii, dset, protein_id, seq_length = item[:6]
                    self.protein_list.append((ii, id_idx, seq_length))
                else:
                    # Handle different formats
                    self.protein_list.append(item[:3] if len(item) >= 3 else (0, 0, 100))
            except Exception as e:
                print(f"Error processing protein index item {item}: {e}")
                continue
        
        self.window_size = window_size
        print(f"Dataset initialized with {len(self.protein_list)} samples, window_size={window_size}")

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get item from dataset with robust error handling."""
        try:
            ii, protein_id, seq_length = self.protein_list[index]
            ii = int(ii)
            protein_id = int(protein_id) if protein_id < len(self.all_encodes) else 0
            seq_length = int(seq_length)
            
            # Get protein sequence features
            all_seq_features = self.all_encodes[protein_id]
            
            # Validate and convert to list if needed
            if isinstance(all_seq_features, torch.Tensor):
                all_seq_features = all_seq_features.cpu().numpy().tolist()
            elif isinstance(all_seq_features, np.ndarray):
                all_seq_features = all_seq_features.tolist()
            elif not isinstance(all_seq_features, list):
                raise ValueError(f"Unexpected all_seq_features type: {type(all_seq_features)}")
            
            # Ensure non-empty features
            if len(all_seq_features) == 0:
                all_seq_features = [[0.0] * 2560]
            
            embedding_dim = len(all_seq_features[0])
            
            # Create local features based on window size
            if self.window_size == 0:
                # No windowing, just use the single residue
                if ii < len(all_seq_features):
                    local_features = [all_seq_features[ii]]
                else:
                    local_features = [[0.0] * embedding_dim]
            else:
                # Windowed features
                local_features = []
                for pos in range(ii - self.window_size, ii + self.window_size + 1):
                    if 0 <= pos < len(all_seq_features):
                        local_features.append(all_seq_features[pos])
                    else:
                        local_features.append([0.0] * embedding_dim)
            
            # Get label
            protein_labels = self.all_label[protein_id]
            if isinstance(protein_labels, np.ndarray):
                protein_labels = protein_labels.tolist()
            
            if ii < len(protein_labels):
                label = float(protein_labels[ii])
            else:
                label = 0.0
            
            # Convert to numpy arrays
            local_features = np.array(local_features, dtype=np.float32)
            all_seq_features = np.array(all_seq_features, dtype=np.float32)
            label = np.array(label, dtype=np.float32)
            
            return local_features, all_seq_features, label
            
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {e}")
            
            # Return safe default values
            embedding_dim = 2560
            local_features = np.array([[0.0] * embedding_dim], dtype=np.float32)
            all_seq_features = np.array([[0.0] * embedding_dim], dtype=np.float32)
            label = np.array(0.0, dtype=np.float32)
            
            return local_features, all_seq_features, label

    def __len__(self) -> int:
        return len(self.protein_list)


def collate_fn(batch):
    """Robust collate function for DataLoader."""
    try:
        local_features, all_seq_features, labels = [], [], []
        
        for local, seq, label in batch:
            local_features.append(local)
            all_seq_features.append(seq)
            labels.append(label)
        
        # Find maximum lengths
        max_local_len = max(local.shape[0] for local in local_features)
        max_protein_len = max(protein.shape[0] for protein in all_seq_features)
        
        batch_size = len(labels)
        local_dim = local_features[0].shape[1]
        protein_dim = all_seq_features[0].shape[1]
        
        # Pad sequences
        padded_locals = np.zeros((batch_size, max_local_len, local_dim), dtype=np.float32)
        padded_proteins = np.zeros((batch_size, max_protein_len, protein_dim), dtype=np.float32)
        padded_labels = np.zeros(batch_size, dtype=np.int64)
        
        local_lengths = []
        protein_lengths = []
        
        for i, (local, protein, label) in enumerate(zip(local_features, all_seq_features, labels)):
            local_len = local.shape[0]
            protein_len = protein.shape[0]
            
            padded_locals[i, :local_len, :] = local
            padded_proteins[i, :protein_len, :] = protein
            padded_labels[i] = int(label)
            
            local_lengths.append(local_len)
            protein_lengths.append(protein_len)
        
        return padded_locals, padded_proteins, padded_labels, local_lengths, protein_lengths
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        
        # Return minimal valid batch
        return (
            np.array([[[0.0] * 2560]], dtype=np.float32),  # [1, 1, 2560]
            np.array([[[0.0] * 2560]], dtype=np.float32),  # [1, 1, 2560]
            np.array([0], dtype=np.int64),                  # [1]
            [1],  # local_lengths
            [1]   # protein_lengths
        )
