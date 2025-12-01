"""
Data preprocessing script for protein sequences.
Converts FASTA files to the required format for training.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import esm


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
    
    # Initialize ESM2 model (esm2_t36_3B_UR50D -> 2560-dim embeddings)
    print("Loading ESM2 model (esm2_t36_3B_UR50D)...")

    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    # Generate embeddings and labels
    embeddings_all_proteins: List[np.ndarray] = []
    labels_all_proteins: List[np.ndarray] = []

    with torch.no_grad():
        for idx, protein in enumerate(proteins):
            name = protein['name']
            seq = protein['sequence']
            seq_len = len(seq)

            if seq_len == 0:
                embeddings_all_proteins.append(np.empty((0, 2560), dtype=np.float32))
                labels_all_proteins.append(np.empty((0,), dtype=np.int32))
                continue

            batch_data = [(name, seq)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            results = model(batch_tokens, repr_layers=[36], return_contacts=False)
            token_representations = results["representations"][36]

            per_residue = token_representations[0, 1:seq_len + 1, :].detach()
            embedding = per_residue.to(dtype=torch.float32).cpu().numpy()

            label_seq = protein['labels']
            labels = np.asarray(
                [1 if label_seq[pos] == '1' else 0 for pos in range(seq_len)],
                dtype=np.int32
            )

            embeddings_all_proteins.append(embedding)
            labels_all_proteins.append(labels)

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"Encoded {idx + 1}/{len(proteins)} proteins")
    
    # Save processed data scoped under the requested split directory
    split_dir = output_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem

    encode_file = split_dir / f"{input_stem}-ESM2.pkl"
    label_file = split_dir / f"{input_stem}-label.pkl"
    list_file = split_dir / f"{input_stem}-list.pkl"
    
    with open(encode_file, 'wb') as f:
        pickle.dump(embeddings_all_proteins, f)

    with open(label_file, 'wb') as f:
        pickle.dump(labels_all_proteins, f)
    
    with open(list_file, 'wb') as f:
        pickle.dump(protein_list, f)
    
    print(f"Saved processed data to {split_dir}")
    print(f"- Encodings: {encode_file}")
    print(f"- Labels: {label_file}")
    print(f"- Protein list: {list_file}")


if __name__ == "__main__":
    main()
