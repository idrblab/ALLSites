#!/usr/bin/env python3
"""
Complete training script for protein binding site predictor.

This script handles:
- Single and multi-GPU training
- Data loading and preprocessing
- Model training with validation
- Checkpointing and early stopping
- Comprehensive logging and metrics
"""

import argparse
import os
import sys
import yaml
import pickle
import timeit
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.sampler as sampler
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model import (
    Predictor, Encoder, Decoder, DecoderLayer, 
    SelfAttention, PositionwiseFeedforward, Trainer, Tester
)
from src.data.data_generator import dataSet
from src.utils.helpers import init_seeds, todevice
from src.utils.metrics import metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train protein binding site predictor")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--distributed", action="store_true", 
                       help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=0, 
                       help="Local rank for distributed training")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with smaller dataset")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed() -> Tuple[bool, int, torch.device]:
    """Setup distributed training if available."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        try:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            return True, local_rank, device
        except Exception as e:
            print(f"Failed to initialize distributed training: {e}")
            return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(config: Dict[str, Any], debug: bool = False) -> Tuple[dataSet, dataSet, dataSet]:
    """Load training, validation, and test datasets."""
    print("Loading datasets...")
    
    # File paths from config
    train_encode_file = config['data']['train_path'] + 'Com_Train_1628_ESMFold.pkl'
    train_label_file = config['data']['train_path'] + 'Com_Train_1628_label.pkl'
    train_list_file = config['data']['train_path'] + 'Com_Train_1628_list.pkl'
    
    valid_encode_file = config['data']['valid_path'] + 'Com_Valid_348_ESMFold.pkl'
    valid_label_file = config['data']['valid_path'] + 'Com_Valid_348_label.pkl'
    valid_list_file = config['data']['valid_path'] + 'Com_Valid_348_list.pkl'

    
    window_size = config['data']['window_size']
    
    # Create datasets
    train_dataset = dataSet(window_size, train_encode_file, train_label_file, train_list_file)
    valid_dataset = dataSet(window_size, valid_encode_file, valid_label_file, valid_list_file)
    test_dataset = dataSet(window_size, test_encode_file, test_label_file, test_list_file)
    
    print(f"Loaded datasets - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, valid_dataset, test_dataset


def create_data_loaders(train_dataset: dataSet, valid_dataset: dataSet, test_dataset: dataSet,
                       config: Dict[str, Any], is_distributed: bool) -> Tuple:
    """Create data loaders for training, validation, and testing."""
    
    # Load indices
    with open(config['data']['train_path'] + 'Com_Train_1628_list.pkl', "rb") as fp:
        train_index = pickle.load(fp)
    train_list = [item[0] for item in train_index]
    
    with open(config['data']['valid_path'] + 'Com_Valid_348_list.pkl', "rb") as fp:
        valid_index = pickle.load(fp)
    valid_list = [item[0] for item in valid_index]
    
    with open(config['data']['test_path'] + 'Com_Test_348_list.pkl', "rb") as fp:
        test_index = pickle.load(fp)
    test_list = [item[0] for item in test_index]
    
    # Create samplers
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_list)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_list)
    else:
        train_sampler = sampler.SubsetRandomSampler(train_list)
        valid_sampler = sampler.SubsetRandomSampler(valid_list)
    
    test_sampler = sampler.SubsetRandomSampler(test_list)
    
    # Collate functions
    def stack_fn_train(batch):
        return create_batch(batch, train_dataset, config)
    
    def stack_fn_valid(batch):
        return create_batch(batch, valid_dataset, config)
    
    def stack_fn_test(batch):
        local_features, all_seq_features, labels = [], [], []
        for local, seq, label in batch:
            local_features.append(local)
            all_seq_features.append(seq)
            labels.append(label)
        return pad_sequences(local_features, all_seq_features, labels, config)
    
    batch_size = config['training']['batch_size']
    
    # Create data loaders
    if is_distributed:
        train_loader = torch.utils.data.DataLoader(
            train_list, batch_size=batch_size, sampler=train_sampler,
            num_workers=0, collate_fn=stack_fn_train, drop_last=False
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_list, batch_size=batch_size, sampler=valid_sampler,
            num_workers=0, collate_fn=stack_fn_valid, drop_last=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=0, collate_fn=stack_fn_test, drop_last=False
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=0, collate_fn=stack_fn_test, drop_last=False
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=0, collate_fn=stack_fn_test, drop_last=False
    )
    
    return train_loader, valid_loader, test_loader


def create_batch(batch_indices: List[int], dataset: dataSet, config: Dict[str, Any]) -> Tuple:
    """Create a batch from dataset indices."""
    local_features, all_seq_features, labels = [], [], []
    
    for i in batch_indices:
        local, seq, label = dataset[i]
        local_features.append(local)
        all_seq_features.append(seq)
        labels.append(label)
    
    return pad_sequences(local_features, all_seq_features, labels, config)


def pad_sequences(local_features: List, all_seq_features: List, labels: List, 
                 config: Dict[str, Any]) -> Tuple:
    """Pad sequences to same length within batch."""
    locals_len = max(local.shape[0] for local in local_features)
    proteins_len = max(protein.shape[0] for protein in all_seq_features)
    
    N = len(labels)
    local_num = [local.shape[0] for local in local_features]
    protein_num = [protein.shape[0] for protein in all_seq_features]
    
    local_dim = config['data']['local_dim']
    protein_dim = config['data']['protein_dim']
    
    # Pad local features
    locals_new = np.zeros((N, locals_len, local_dim))
    for i, local in enumerate(local_features):
        a_len = local.shape[0]
        locals_new[i, :a_len, :] = local
    
    # Pad protein features
    proteins_new = np.zeros((N, proteins_len, protein_dim))
    for i, protein in enumerate(all_seq_features):
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = protein
    
    # Process labels
    labels_new = np.zeros(N, dtype=np.int64)
    for i, label in enumerate(labels):
        labels_new[i] = label
    
    return (locals_new, proteins_new, labels_new, local_num, protein_num)

def create_model(config: Dict[str, Any], device: torch.device) -> Predictor:
    """Create the protein binding site predictor model."""
    print("Creating model...")
    
    # Model parameters
    protein_dim = config['model']['protein_dim']
    local_dim = config['model']['local_dim']
    hid_dim = config['model']['hidden_dim']
    n_layers = config['model']['n_layers']
    n_heads = config['model']['n_heads']
    pf_dim = config['model']['pf_dim']
    dropout = config['model']['dropout']
    kernel_size = config['model']['kernel_size']
    
    # Create encoder and decoder
    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, 
                     SelfAttention, PositionwiseFeedforward, dropout, device)
    
    # Create complete model
    model = Predictor(encoder, decoder, device)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    return model


def setup_output_directories(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Setup output directories for models and results."""
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_dir'])
    result_dir = Path(config['paths']['result_dir'])
    
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    # Create experiment-specific files
    experiment_name = "Com-Train1628-Val348-Test348"
    file_AUCs = result_dir / f"output-{experiment_name}.txt"
    file_model = model_dir / f"{experiment_name}-best"
    file_model_test = model_dir / f"{experiment_name}-test-best"
    
    return str(file_AUCs), str(file_model), str(file_model_test)


def train_epoch(trainer: Trainer, train_loader, device: torch.device, 
               is_distributed: bool) -> float:
    """Train model for one epoch."""
    return trainer.train(train_loader, device)


def validate_epoch(tester: Tester, valid_loader, device: torch.device, 
                  is_distributed: bool) -> Tuple:
    """Validate model for one epoch."""
    correct_labels, predicted_labels, predicted_scores = tester.test(valid_loader, device)
    
    if is_distributed:
        # Gather results from all processes
        correct_labels = torch.tensor(np.array(correct_labels), dtype=torch.float64, device='cuda')
        predicted_labels = torch.tensor(np.array(predicted_labels), dtype=torch.float64, device='cuda')
        predicted_scores = torch.tensor(np.array(predicted_scores), dtype=torch.float64, device='cuda')
        
        # All gather
        correct_labels_list = [torch.zeros_like(correct_labels).cuda() for _ in range(dist.get_world_size())]
        predicted_labels_list = [torch.zeros_like(predicted_labels).cuda() for _ in range(dist.get_world_size())]
        predicted_scores_list = [torch.zeros_like(predicted_scores).cuda() for _ in range(dist.get_world_size())]
        
        dist.all_gather(correct_labels_list, correct_labels)
        dist.all_gather(predicted_labels_list, predicted_labels)
        dist.all_gather(predicted_scores_list, predicted_scores)
        
        # Concatenate results
        correct_labels = torch.cat(correct_labels_list, dim=0).cpu().numpy()
        predicted_labels = torch.cat(predicted_labels_list, dim=0).cpu().numpy()
        predicted_scores = torch.cat(predicted_scores_list, dim=0).cpu().numpy()
        
        dist.barrier()
    
    return correct_labels, predicted_labels, predicted_scores


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Setup distributed training
    is_distributed, local_rank, device = setup_distributed()
    
    if is_distributed:
        print(f"Distributed training enabled. Local rank: {local_rank}, World size: {dist.get_world_size()}")
    else:
        print(f"Single process training on device: {device}")
    
    # Initialize seeds for reproducibility
    init_seeds(config['training']['seed'])
    
    # Load datasets
    train_dataset, valid_dataset, test_dataset = load_data(config, args.debug)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_dataset, valid_dataset, test_dataset, config, is_distributed
    )
    
    # Create model
    model = create_model(config, device)
    
    # Wrap model for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                   find_unused_parameters=True)
    
    # Create trainer and tester
    trainer = Trainer(model, config['training']['learning_rate'], 
                     config['training']['weight_decay'])
    tester = Tester(model)
    
    # Setup output directories and files
    file_AUCs, file_model, file_model_test = setup_output_directories(config)
    
    # Initialize metrics logging
    if not is_distributed or dist.get_rank() == 0:
        header = ('Epoch\tTime1(sec)\tTime2(sec)\tLoss_train\tACC_dev\tAUC_dev\t'
                 'Rec_dev\tPre_dev\tF1_dev\tMCC_dev\tPRC_dev\tACC_test\tAUC_test\t'
                 'Rec_test\tPre_test\tF1_test\tMCC_test\tPRC_test')
        with open(file_AUCs, 'w') as f:
            f.write(header + '\n')
        print('Training started...')
        print(header)
    
    # Training parameters
    max_PRC_dev = 0
    max_PRC_test = 0
    last_improve = 0
    epochs = config['training']['epochs']
    early_stopping = config['training']['early_stopping']
    decay_interval = config['training']['decay_interval']
    lr_decay = config['training']['lr_decay']
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Set epoch for distributed sampler
        if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # Learning rate decay
        if epoch % decay_interval == 0:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= lr_decay
        
        # Training phase
        start_time = timeit.default_timer()
        loss_train = train_epoch(trainer, train_loader, device, is_distributed)
        train_time = timeit.default_timer() - start_time
        
        # Validation phase
        eval_start = timeit.default_timer()
        correct_labels_valid, predicted_labels_valid, predicted_scores_valid = validate_epoch(
            tester, valid_loader, device, is_distributed
        )
        
        # Calculate validation metrics
        ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev = metrics(
            correct_labels_valid, predicted_labels_valid, predicted_scores_valid
        )
        
        # Test phase
        correct_labels_test, predicted_labels_test, predicted_scores_test = tester.test(test_loader, device)
        ACC_test, AUC_test, Rec_test, Pre_test, F1_test, MCC_test, PRC_test = metrics(
            correct_labels_test, predicted_labels_test, predicted_scores_test
        )
        
        eval_time = timeit.default_timer() - eval_start
        
        # Prepare metrics for logging
        epoch_metrics = [
            epoch, train_time, eval_time, loss_train,
            ACC_dev, AUC_dev, Rec_dev, Pre_dev, F1_dev, MCC_dev, PRC_dev,
            ACC_test, AUC_test, Rec_test, Pre_test, F1_test, MCC_test, PRC_test
        ]
        
        # Save metrics (only on main process for distributed training)
        if not is_distributed or dist.get_rank() == 0:
            tester.save_AUCs(epoch_metrics, file_AUCs)
            print('\t'.join(map(str, [f'{x:.4f}' if isinstance(x, float) else str(x) 
                                     for x in epoch_metrics])))
        
        # Model checkpointing based on validation PRC
        if PRC_dev > max_PRC_dev:
            last_improve = epoch
            max_PRC_dev = PRC_dev
            
            if not is_distributed or dist.get_rank() == 0:
                print(f'Validation improved at epoch {last_improve}, saving model...')
                tester.save_model(model, file_model)
        
        
        # Early stopping check
        if epoch - last_improve >= early_stopping:
            if not is_distributed or dist.get_rank() == 0:
                print(f'Early stopping at epoch {epoch} (no improvement for {early_stopping} epochs)')
            break
    
    # Training completed
    if not is_distributed or dist.get_rank() == 0:
        print('Training completed!')
        print(f'Best validation PRC: {max_PRC_dev:.4f}')
        print(f'Best test PRC: {max_PRC_test:.4f}')
        print(f'Models saved to: {file_model}')
    
    # Cleanup for distributed training
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
