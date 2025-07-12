"""
Data preprocessing script for protein sequences.
Converts FASTA files to the required format for training.
"""

import argparse
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, choices=['train', 'valid', 'test'], 
                       required=True, help="Data split type")
    return parser.parse_args()


def parse_fasta_file(fasta_path: str) -> List[Dict]:
    """Parse FASTA file with protein sequences and labels."""
    proteins = []
    
    with open(fasta_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('>'):
            name = lines[i][1:]  # Remove '>'
            sequence = lines[i + 1] if i + 1 < len(lines) else ''
            labels = lines[i + 2] if i + 2 < len(lines) else ''
            
            proteins.append({
                'name': name,
                'sequence': sequence,
                'labels': labels
            })
            i += 3
        else:
            i += 1
    
    return proteins


def create_protein_list(proteins: List[Dict]) -> List[Tuple]:
    """Create protein list for dataset."""
    protein_list = []
    
    for protein_idx, protein in enumerate(proteins):
        sequence = protein['sequence']
        name = protein['name']
        seq_length = len(sequence)
        
        for residue_idx in range(seq_length):
            protein_list.append((
                len(protein_list),  # count
                protein_idx,        # id_idx
                residue_idx,        # ii (position in sequence)
                'processed',        # dset
                name,              # protein_id
                seq_length         # seq_length
            ))
    
    return protein_list


def main():
    args = parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse input file
    proteins = parse_fasta_file(args.input)
    print(f"Parsed {len(proteins)} proteins")
    
    # Create protein list
    protein_list = create_protein_list(proteins)
    
    # Note: This is a template - actual ESMFold encoding would happen here
    # For now, create dummy embeddings as placeholders
    dummy_embeddings = []
    dummy_labels = []
    
    for protein in proteins:
        seq_len = len(protein['sequence'])
        # Placeholder: replace with actual ESMFold encoding
        embedding = [[0.0] * 2560 for _ in range(seq_len)]
        labels = [int(c) for c in protein['labels'][:seq_len]]
        
        dummy_embeddings.append(embedding)
        dummy_labels.append(labels)
    
    # Save processed data
    split_name = args.split.capitalize()
    
    encode_file = output_dir / f"Com_{split_name}_{len(proteins)}_ESMFold.pkl"
    label_file = output_dir / f"Com_{split_name}_{len(proteins)}_label.pkl"
    list_file = output_dir / f"Com_{split_name}_{len(proteins)}_list.pkl"
    
    with open(encode_file, 'wb') as f:
        pickle.dump(dummy_embeddings, f)
    
    with open(label_file, 'wb') as f:
        pickle.dump(dummy_labels, f)
    
    with open(list_file, 'wb') as f:
        pickle.dump(protein_list, f)
    
    print(f"Saved processed data to {output_dir}")
    print(f"- Encodings: {encode_file}")
    print(f"- Labels: {label_file}")
    print(f"- Protein list: {list_file}")


if __name__ == "__main__":
    main()