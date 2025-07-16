"""
Main model architecture for protein binding site prediction.

This module contains the core neural network components including:
- SelfAttention: Multi-head attention mechanism
- Encoder: Convolutional protein feature extractor  
- Decoder: Transformer decoder with attention
- Predictor: Complete model combining encoder-decoder
- Trainer/Tester: Training and evaluation classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import timeit
from typing import Tuple, Optional, List

try:
    from radam import RAdam
    from lookahead import Lookahead
except ImportError:
    try:
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        sys.path.append(str(parent_dir))
        from optimizers.radam import RAdam
        from optimizers.lookahead import Lookahead
    except ImportError:
        from ..optimizers.radam import RAdam
        from ..optimizers.lookahead import Lookahead


class SelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        
        assert hid_dim % n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hid_dim]
            key: Key tensor [batch_size, seq_len, hid_dim]
            value: Value tensor [batch_size, seq_len, hid_dim]
            mask: Optional attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, hid_dim]
            attention: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        
        # Reshape back
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        
        return x, attention


class Encoder(nn.Module):
    """Convolutional encoder for protein feature extraction."""
    
    def __init__(self, protein_dim: int, hid_dim: int, n_layers: int, 
                 kernel_size: int, dropout: float, device: torch.device):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([
            nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(protein_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
    
    def forward(self, protein: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of encoder.
        
        Args:
            protein: Protein embeddings [batch_size, seq_len, protein_dim]
            
        Returns:
            Encoded features [batch_size, seq_len, hid_dim]
        """
        # Linear projection
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)  # [batch_size, hid_dim, seq_len]
        
        # Convolutional layers with GLU activation
        for conv in self.convs:
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)  # Gated Linear Unit
            conved = (conved + conv_input) * self.scale  # Residual connection
            conv_input = conved
        
        # Back to original shape and normalize
        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        
        return conved


class PositionwiseFeedforward(nn.Module):
    """Position-wise feedforward network."""
    
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feedforward network."""
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""
    
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, 
                 self_attention, positionwise_feedforward, dropout: float, device: torch.device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)  # Self-attention
        self.ea = self_attention(hid_dim, n_heads, dropout, device)  # Cross-attention
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)
    
    def forward(self, trg: torch.Tensor, src: torch.Tensor, 
                trg_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder layer.
        
        Args:
            trg: Target sequence [batch_size, tgt_len, hid_dim]
            src: Source sequence [batch_size, src_len, hid_dim]
            trg_mask: Target mask
            src_mask: Source mask
            
        Returns:
            output: Layer output [batch_size, tgt_len, hid_dim]
            attention: Cross-attention weights
        """
        # Self-attention
        trg_1 = trg
        trg, _ = self.sa(trg, trg, trg, trg_mask)
        trg = self.ln(trg_1 + self.do(trg))
        
        # Cross-attention
        trg_2 = trg
        trg, attention = self.ea(trg, src, src, src_mask)
        trg = self.ln(trg_2 + self.do(trg))
        
        # Feedforward
        trg_3 = trg
        trg = self.ln(trg_3 + self.do(self.pf(trg)))
        
        return trg, attention


class Decoder(nn.Module):
    """Transformer decoder for binding site prediction."""
    
    def __init__(self, local_dim: int, hid_dim: int, n_layers: int, n_heads: int, 
                 pf_dim: int, decoder_layer, self_attention, positionwise_feedforward, 
                 dropout: float, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.device = device
        
        self.layers = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention, 
                         positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        
        self.ft = nn.Linear(local_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        
        # Classification head
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 2)
    
    def forward(self, trg: torch.Tensor, src: torch.Tensor, 
                trg_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder.
        
        Args:
            trg: Local features [batch_size, local_len, local_dim]
            src: Encoded protein features [batch_size, protein_len, hid_dim]
            trg_mask: Target mask
            src_mask: Source mask
            
        Returns:
            sum_features: Pooled features [batch_size, hid_dim]
            attention: Final attention weights
            logits: Classification logits [batch_size, 2]
        """
        # Project local features to hidden dimension
        trg = self.ft(trg)
        
        # Pass through decoder layers
        for layer in self.layers:
            trg, attention = layer(trg, src, trg_mask, src_mask)
        
        # Attention-based pooling
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        
        sum_features = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, :] * norm[i, j]
                sum_features[i, :] += v
        
        # Classification head
        logits = F.relu(self.fc_1(sum_features))
        logits = self.do(logits)
        logits = F.relu(self.fc_2(logits))
        logits = self.fc_3(logits)
        
        return sum_features, attention, logits


class Predictor(nn.Module):
    """Complete protein binding site predictor model."""
    
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def make_masks(self, local_num: List[int], protein_num: List[int], 
                   local_max_len: int, protein_max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create attention masks for variable-length sequences."""
        N = len(local_num)
        local_mask = torch.zeros((N, local_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        
        for i in range(N):
            local_mask[i, :local_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        
        local_mask = local_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        
        return local_mask, protein_mask
    
    def forward(self, local: torch.Tensor, protein: torch.Tensor, 
                local_num: List[int], protein_num: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of complete model.
        
        Args:
            local: Local features [batch_size, local_max_len, local_dim]
            protein: Protein features [batch_size, protein_max_len, protein_dim]
            local_num: Actual lengths of local sequences
            protein_num: Actual lengths of protein sequences
            
        Returns:
            sum_features: Pooled features
            attention: Attention weights
            logits: Classification logits
        """
        local_max_len = local.shape[1]
        protein_max_len = protein.shape[1]
        local_mask, protein_mask = self.make_masks(local_num, protein_num, local_max_len, protein_max_len)
        
        # Encode protein features
        enc_src = self.encoder(protein)
        
        # Decode and predict
        sum_features, attention, logits = self.decoder(local, enc_src, local_mask, protein_mask)
        
        return sum_features, attention, logits
    
    def __call__(self, data: Tuple, train: bool = True):
        """Model call for training/inference."""
        local, protein, correct_interaction, local_num, protein_num = data
        Loss = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float().to(self.device))
        
        if train:
            sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
            del sum_features, attention  # Memory cleanup
            loss = Loss(predicted_interaction, correct_interaction)
            return loss
        else:
            sum_features, attention, predicted_interaction = self.forward(local, protein, local_num, protein_num)
            del sum_features, attention  # Memory cleanup
            
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            
            return correct_labels, predicted_labels, predicted_scores


class Trainer:
    """Training class for the protein binding site predictor - 简化版本"""
    
    def __init__(self, model: Predictor, lr: float, weight_decay: float):
        self.model = model
        
        # Separate weight and bias parameters for different regularization
        weight_p, bias_p = [], []
        
        # Initialize parameters
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Group parameters
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        
        self.optimizer = RAdam([
            {'params': weight_p, 'weight_decay': weight_decay}, 
            {'params': bias_p, 'weight_decay': 0}
        ], lr=lr)

        # self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
    
    def train(self, dataloader, device: torch.device) -> float:
        """Train the model for one epoch."""
        self.model.train()
        loss_total = 0
        
        for batch_idx, (local, protein, label, local_num, protein_num) in enumerate(dataloader):
            print(f"Batch {batch_idx}")
            
            torch.cuda.empty_cache()
            
            # Move data to device with improved handling
            try:
                from utils.helpers import todevice
                data_pack = todevice(local, protein, label, local_num, protein_num, device)
            except ImportError:
                if isinstance(local, np.ndarray):
                    local = torch.from_numpy(local).float().to(device)
                else:
                    local = local.float().to(device)
                
                if isinstance(protein, np.ndarray):
                    protein = torch.from_numpy(protein).float().to(device)
                else:
                    protein = protein.float().to(device)
                    
                if isinstance(label, np.ndarray):
                    label = torch.from_numpy(label).long().to(device)
                else:
                    label = label.long().to(device)
                
                if local.dim() == 4 and local.shape[0] == 1:
                    local = local.squeeze(0)
                if protein.dim() == 4 and protein.shape[0] == 1:
                    protein = protein.squeeze(0)
                if label.dim() == 2 and label.shape[0] == 1:
                    label = label.squeeze(0)
                
                print(f"Data shapes - Local: {local.shape}, Protein: {protein.shape}, Label: {label.shape}")
                data_pack = (local, protein, label, local_num, protein_num)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            loss = self.model(data_pack)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            loss_total += loss.item()
            del data_pack
        
        return loss_total


class Tester:
    """Testing/evaluation class for the protein binding site predictor."""
    
    def __init__(self, model: Predictor):
        self.model = model
    
    def test(self, dataloader, device: torch.device) -> Tuple[List, List, List]:
        """Evaluate the model on test data."""
        self.model.eval()
        T, Y, S = [], [], []
        
        with torch.no_grad():
            for batch_idx, (local, protein, label, local_num, protein_num) in enumerate(dataloader):
                print(f"Test batch {batch_idx}")
                
                torch.cuda.empty_cache()
                
                # Move data to device
                try:
                    from utils.helpers import todevice
                    data_pack = todevice(local, protein, label, local_num, protein_num, device)
                except ImportError:
                    local = torch.from_numpy(local).float().to(device) if isinstance(local, np.ndarray) else local.to(device)
                    protein = torch.from_numpy(protein).float().to(device) if isinstance(protein, np.ndarray) else protein.to(device)
                    label = torch.from_numpy(label).long().to(device) if isinstance(label, np.ndarray) else label.to(device)
                    data_pack = (local, protein, label, local_num, protein_num)
                
                # Get predictions
                correct_labels, predicted_labels, predicted_scores = self.model(data_pack, train=False)
                
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
                
                del data_pack
        
        return T, Y, S
    
    def save_AUCs(self, AUCs: List, filename: str):
        """Save evaluation metrics to file."""
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')
    
    def save_model(self, model: nn.Module, filename: str):
        """Save model state dict."""
        if hasattr(model, 'module'):  # For DDP models
            torch.save(model.module.state_dict(), filename)
        else:
            torch.save(model.state_dict(), filename)
