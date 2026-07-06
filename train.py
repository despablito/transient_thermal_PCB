"""
Training Script for Transient Thermal Neural Operator
======================================================
Autoregressive training with physics-informed loss and synthetic cooling drill.

Loss = LogCosh(ΔT_pred, ΔT_true) + PhysicsDrill + CoolingConstraint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time
import logging
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import RollingWindowThermalUNet, create_model


def _ema_smooth(values, alpha=0.6):
    """Exponential moving average for plot smoothing."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * smoothed[-1] + (1 - alpha) * v)
    return smoothed


def plot_learning_curve(history: Dict, save_path: str = 'visualizations/learning_curve.png'):
    """Plot and save learning curves after each epoch."""
    if len(history['train_loss']) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    train_raw = history['train_loss']
    val_raw = history['val_loss']
    train_smooth = _ema_smooth(train_raw)
    val_smooth = _ema_smooth(val_raw)
    
    # Panel 1: Train vs Val Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_raw, 'b-', alpha=0.25, linewidth=1)
    ax1.plot(epochs, val_raw, 'r-', alpha=0.25, linewidth=1)
    ax1.plot(epochs, train_smooth, 'b-', linewidth=2.5, label='Train Loss')
    ax1.plot(epochs, val_smooth, 'r-', linewidth=2.5, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Correction RMS growth
    ax2 = axes[1]
    creg = history.get('correction_reg', [])
    corr_mag = [c ** 0.5 * 300.0 for c in creg] if creg else []
    if corr_mag:
        ax2.plot(epochs, corr_mag, 'darkorange', linewidth=2.5, label='Correction RMS [°C]')
        ax2.fill_between(epochs, 0, corr_mag, color='darkorange', alpha=0.15)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Neural Correction RMS [°C]')
    ax2.set_title('Neural Correction Magnitude')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: LR schedule
    ax3 = axes[2]
    ax3.plot(epochs, history['lr'], 'g-', linewidth=2.5)
    ax3.fill_between(epochs, 0, history['lr'], color='green', alpha=0.1)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(-4,-4))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_file: str = 'training.log') -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger('ThermalTrainer')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    data_pattern: str = 'checkpoint_*.npz'
    train_split: float = 0.8
    val_split: float = 0.1
    # Seed for the train/val/test split. MUST be identical at training and
    # evaluation time — otherwise the "test" set of a later re-split overlaps
    # the training set and benchmark numbers are contaminated.
    split_seed: int = 42
    
    # Optimization
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    
    # DataLoader (num_workers=0 avoids macOS semaphore leaks)
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    
    # Sequence handling
    sequence_length: int = 20
    teacher_forcing_start: float = 0.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_decay_epochs: int = 1
    
    # Loss weights
    physics_loss_weight: float = 1.0
    correction_reg_weight: float = 0.0
    
    # Normalization scales (derived from data range)
    P_scale: float = 1e10       # [W/m³]
    T_scale: float = 300.0      # [°C]
    h_scale: float = 20000.0    # [W/(m²·K)]
    
    # Device
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    use_amp: bool = False
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 5


class ThermalSequenceDataset(Dataset):
    """
    Dataset for transient thermal sequences.
    
    Loads pre-generated FDM data and prepares rolling window inputs.
    """
    
    def __init__(self, data: Dict[str, np.ndarray], indices: np.ndarray,
                 config: TrainingConfig, sequence_length: int = 20):
        """
        Args:
            data: Dictionary from npz file
            indices: Sequence indices to use (train/val/test split)
            config: Training configuration
            sequence_length: Number of timesteps per sample
        """
        self.T_maps = data['T_maps'][indices]    # [N, T, H, W]
        self.P_maps = data['P_maps'][indices]    # [N, T, H, W]
        self.h_values = data['h_values'][indices]       # [N]
        self.T_amb_values = data['T_amb_values'][indices]  # [N]
        
        self.config = config
        self.sequence_length = sequence_length
        self.n_sequences = len(indices)
        self.n_frames = self.T_maps.shape[1]
        
        # Number of valid starting positions per sequence
        self.starts_per_seq = max(1, self.n_frames - sequence_length)
        
    def __len__(self) -> int:
        return self.n_sequences * self.starts_per_seq
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map flat index to (sequence, start_frame)
        seq_idx = idx // self.starts_per_seq
        start_frame = idx % self.starts_per_seq
        
        # Extract subsequence
        end_frame = start_frame + self.sequence_length
        
        T_seq = self.T_maps[seq_idx, start_frame:end_frame]  # [T, H, W]
        P_seq = self.P_maps[seq_idx, start_frame:end_frame]  # [T, H, W]
        
        # Get initial temperature (frame before start)
        if start_frame > 0:
            T_initial = self.T_maps[seq_idx, start_frame - 1]
            P_initial = self.P_maps[seq_idx, start_frame - 1]
        else:
            T_initial = np.full_like(T_seq[0], self.T_amb_values[seq_idx])
            P_initial = np.zeros_like(P_seq[0])
        
        # Get operating conditions
        h = self.h_values[seq_idx]
        T_amb = self.T_amb_values[seq_idx]
        
        # Convert to tensors and add channel dimension
        return {
            'T_seq': torch.from_numpy(T_seq).unsqueeze(1).float(),      # [T, 1, H, W]
            'P_seq': torch.from_numpy(P_seq).unsqueeze(1).float(),      # [T, 1, H, W]
            'T_initial': torch.from_numpy(T_initial).unsqueeze(0).float(),  # [1, H, W]
            'P_initial': torch.from_numpy(P_initial).unsqueeze(0).float(),  # [1, H, W]
            'h': torch.tensor(h).float(),
            'T_amb': torch.tensor(T_amb).float(),
        }


class LogCoshLoss(nn.Module):
    """
    Log-Cosh loss function (numerically stable).
    
    L(y, ŷ) = log(cosh(y - ŷ))
    
    Behaves like L1 for large errors, L2 for small errors.
    More robust to outliers than MSE.
    
    For numerical stability: log(cosh(x)) ≈ |x| - log(2) for large |x|
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            weights: Optional spatial weights [B, C, H, W]
        
        Returns:
            Weighted log-cosh loss
        """
        diff = pred - target
        # Numerically stable log-cosh: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        # For large |x|, this avoids overflow in cosh
        abs_diff = torch.abs(diff)
        loss = abs_diff + torch.log1p(torch.exp(-2 * abs_diff)) - 0.6931471805599453  # log(2)
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()


class PhysicsInformedLoss(nn.Module):
    """
    Aggressive Physics-Informed Constraint Loss (v2 - "Physics Drill").
    
    Enforces Newton's Law of Cooling:
    When P is low and T > T_amb, temperature MUST decrease (ΔT < 0).
    
    Key changes from v1:
    - Removed hard threshold (1e-6) that killed gradients
    - Uses soft power threshold with gradient flow
    - Penalizes ΔT >= 0 (not just ΔT > 0) to force cooling
    """
    
    def __init__(self, penalty_strength: float = 1.0, power_threshold: float = 0.05):
        super().__init__()
        self.penalty_strength = penalty_strength
        self.power_threshold = power_threshold  # Normalized power threshold
        
    def forward(self, delta_T: torch.Tensor, P_t: torch.Tensor,
                T_prev: torch.Tensor, T_amb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_T: Predicted temperature change [B, 1, H, W] (NORMALIZED)
            P_t: Current power [B, 1, H, W] (NORMALIZED)
            T_prev: Previous temperature [B, 1, H, W] (NORMALIZED)
            T_amb: Ambient temperature [B, 1, H, W] (NORMALIZED)
        
        Returns:
            Physics violation penalty (differentiable)
        """
        # Soft condition 1: Power is low (smooth transition, not hard cutoff)
        # power_factor = 1.0 when P=0, decays smoothly as P increases
        power_factor = torch.exp(-10.0 * P_t.abs() / self.power_threshold)
        
        # Condition 2: Temperature excess above ambient (how much should it cool)
        # The bigger the difference, the stronger the expected cooling
        temp_excess = F.relu(T_prev - T_amb)  # [0, inf)
        
        # Condition 3: Violation = temperature not decreasing (ΔT >= 0)
        # F.relu(delta_T + epsilon) forces ΔT to be negative
        # Adding small epsilon to penalize ΔT=0 as well (should be cooling!)
        epsilon = 0.001  # Small offset to ensure ΔT < 0, not ΔT <= 0
        violation = F.relu(delta_T + epsilon)  # Penalty for ΔT >= -epsilon
        
        # Combined penalty: stronger when P is low and T >> T_amb
        # This creates gradient signal even when no exact "P=0" samples exist
        penalty = power_factor * temp_excess * violation
        
        return self.penalty_strength * penalty.mean()


class ThermalTrainer:
    """
    Trainer for the transient thermal neural operator.
    
    Handles:
    - Teacher forcing with curriculum learning
    - Physics-informed loss
    - Mixed precision training
    - Checkpointing and logging
    """
    
    def __init__(self, model: RollingWindowThermalUNet, config: TrainingConfig, 
                 logger: Optional[logging.Logger] = None):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        self.logger = logger or logging.getLogger('ThermalTrainer')
        
        # Loss functions
        self.logcosh_loss = LogCoshLoss()
        self.physics_loss = PhysicsInformedLoss(penalty_strength=1.0)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp and config.device == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'physics_loss': [], 'drill_loss': [], 'correction_reg': [], 'lr': [], 'tf_ratio': []}
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        
    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Calculate teacher forcing ratio with linear decay."""
        decay_progress = min(1.0, epoch / self.config.teacher_forcing_decay_epochs)
        ratio = self.config.teacher_forcing_start - (
            self.config.teacher_forcing_start - self.config.teacher_forcing_end
        ) * decay_progress
        return ratio
    
    def compute_temperature_weights(self, T_target: torch.Tensor) -> torch.Tensor:
        """
        Compute sample weights based on temperature.
        Higher weights for higher temperatures (junction hotspots).
        
        Weight = 1 + T_norm * 2 (max weight = 3 for hottest regions)
        """
        # Normalize temperature to [0, 1] range (25-750°C actual range)
        T_norm = (T_target - 25) / 725  # Based on actual data range
        weights = 1.0 + T_norm.clamp(0, 1) * 2.0  # Weight from 1.0 to 3.0
        return weights
    
    def normalize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize batch inputs."""
        cfg = self.config
        
        return {
            'T_seq': batch['T_seq'] / cfg.T_scale,
            'P_seq': batch['P_seq'] / cfg.P_scale,
            'T_initial': batch['T_initial'] / cfg.T_scale,
            'P_initial': batch['P_initial'] / cfg.P_scale,
            'h': batch['h'] / cfg.h_scale,
            'T_amb': batch['T_amb'] / cfg.T_scale,
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor],
                   teacher_forcing_ratio: float,
                   is_training: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with optional teacher forcing and Physics Drill.
        
        Args:
            batch: Normalized input batch
            teacher_forcing_ratio: Probability of using ground truth T_{t-1}
            is_training: If True, applies Physics Drill with synthetic cooling data
        
        Returns:
            Total loss and loss components dictionary
        """
        B, T, _, H, W = batch['T_seq'].shape
        device = self.device
        
        # Move batch to device
        T_seq = batch['T_seq'].to(device)      # [B, T, 1, H, W]
        P_seq = batch['P_seq'].to(device)      # [B, T, 1, H, W]
        T_initial = batch['T_initial'].to(device)  # [B, 1, H, W]
        P_initial = batch['P_initial'].to(device)  # [B, 1, H, W]
        h = batch['h'].to(device).view(B, 1, 1, 1).expand(B, 1, H, W)
        T_amb = batch['T_amb'].to(device).view(B, 1, 1, 1).expand(B, 1, H, W)
        
        total_data_loss = 0.0
        total_physics_loss = 0.0
        total_correction_reg = 0.0
        
        T_prev = T_initial
        P_prev = P_initial
        
        for t in range(T):
            P_t = P_seq[:, t]
            T_target = T_seq[:, t]
            
            # Forward pass
            delta_T_pred = self.model(P_t, P_prev, T_prev, T_amb, h)
            T_pred = T_prev + delta_T_pred
            
            # Compute target delta_T
            delta_T_target = T_target - T_prev
            
            # Temperature-weighted log-cosh loss
            weights = self.compute_temperature_weights(T_target * self.config.T_scale)
            data_loss = self.logcosh_loss(delta_T_pred, delta_T_target, weights)
            
            # L2 regularization on neural correction (keep corrections small)
            delta_T_physics = self.model.physics_residual(P_t, T_prev, T_amb, h)
            correction = delta_T_pred - delta_T_physics
            correction_reg = correction.pow(2).mean()
            
            # Physics-informed constraint
            physics_loss = self.physics_loss(
                delta_T_pred, P_t, T_prev, T_amb
            )
            
            total_data_loss = total_data_loss + data_loss
            total_physics_loss = total_physics_loss + physics_loss
            total_correction_reg = total_correction_reg + correction_reg
            
            # Update for next step (teacher forcing or autoregressive)
            P_prev = P_t
            
            if torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                T_prev = T_target
            else:
                # Autoregressive: use prediction
                T_prev = T_pred.detach()  # Detach to prevent backprop through time explosion
        
        # Average over sequence
        data_loss = total_data_loss / T
        physics_loss = total_physics_loss / T
        correction_reg = total_correction_reg / T
        
        # =====================================================================
        # PHYSICS DRILL: Synthetic cooling scenarios to enforce thermodynamics
        # =====================================================================
        drill_loss = torch.tensor(0.0, device=device)
        
        if is_training:
            # Create synthetic "cooling test" scenarios
            # Scenario: Hot chip (T=150-300°C), zero power, should COOL DOWN
            
            # Synthetic batch (same size as real batch for efficiency)
            P_zero = torch.zeros(B, 1, H, W, device=device)  # No power
            P_zero_prev = torch.zeros(B, 1, H, W, device=device)  # No power before
            
            # Random high temperatures (normalized): 0.3 to 0.8 (90-240°C actual)
            T_hot = (0.3 + 0.5 * torch.rand(B, 1, H, W, device=device))
            
            # Add spatial variation (hotspot in center)
            y = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1)
            x = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W)
            hotspot = torch.exp(-(x**2 + y**2) / 0.5)  # Gaussian hotspot
            T_hot = T_hot * (0.7 + 0.3 * hotspot)  # 70% base + 30% hotspot
            
            # Ambient temperature (normalized): ~0.1 (30°C actual)
            T_amb_synth = torch.full((B, 1, H, W), 0.1, device=device)
            
            # h values (normalized): use same as batch
            h_synth = h.clone()
            
            # Forward pass with synthetic data
            delta_T_synth = self.model(P_zero, P_zero_prev, T_hot, T_amb_synth, h_synth)
            
            # Physics loss on synthetic data (should produce negative ΔT!)
            drill_loss = self.physics_loss(
                delta_T_synth,
                P_zero,
                T_hot,
                T_amb_synth
            )
            
            # Apply strong multiplier (10x) to make physics matter
            drill_loss = drill_loss * 10.0
        
        # Total physics loss = from data + from drill
        total_physics_loss_combined = physics_loss + drill_loss
        
        # Total loss = data + physics + correction regularization
        total_loss = (data_loss 
                      + self.config.physics_loss_weight * total_physics_loss_combined
                      + self.config.correction_reg_weight * correction_reg)
        
        return total_loss, {
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'drill_loss': drill_loss.item() if isinstance(drill_loss, torch.Tensor) else drill_loss,
            'correction_reg': correction_reg.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with Physics Drill."""
        self.model.train()
        
        teacher_forcing_ratio = self.get_teacher_forcing_ratio(self.current_epoch)
        
        epoch_losses = {'data_loss': 0, 'physics_loss': 0, 'drill_loss': 0, 'correction_reg': 0, 'total_loss': 0}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            batch = self.normalize_batch(batch)
            
            self.optimizer.zero_grad()
            
            # is_training=True enables Physics Drill
            loss, loss_dict = self.train_step(batch, teacher_forcing_ratio, is_training=True)
            
            # Backward pass with mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'drill': f"{loss_dict['drill_loss']:.4f}",
                'tf': f"{teacher_forcing_ratio:.2f}"
            })
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model (fully autoregressive, no teacher forcing, no drill)."""
        self.model.eval()
        
        val_losses = {'data_loss': 0, 'physics_loss': 0, 'drill_loss': 0, 'correction_reg': 0, 'total_loss': 0}
        n_batches = 0
        
        for batch in val_loader:
            batch = self.normalize_batch(batch)
            
            # Validation: fully autoregressive, no Physics Drill (is_training=False)
            _, loss_dict = self.train_step(batch, teacher_forcing_ratio=0.0, is_training=False)
            
            for k, v in loss_dict.items():
                if k not in val_losses:
                    val_losses[k] = 0
                val_losses[k] += v
            n_batches += 1
        
        for k in val_losses:
            val_losses[k] /= n_batches
        
        return val_losses
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop."""
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Epochs: {self.config.num_epochs}")
        self.logger.info("-" * 70)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            self.history['physics_loss'].append(train_losses['physics_loss'])
            self.history['drill_loss'].append(train_losses['drill_loss'])
            self.history['correction_reg'].append(train_losses.get('correction_reg', 0))
            self.history['lr'].append(self.scheduler.get_last_lr()[0])
            tf_ratio = self.get_teacher_forcing_ratio(epoch)
            self.history['tf_ratio'].append(tf_ratio)
            
            epoch_time = time.time() - epoch_start
            is_best = val_losses['total_loss'] < self.best_val_loss
            best_marker = " ⭐" if is_best else ""
            
            self.logger.info(
                f"Epoch {epoch + 1:03d}/{self.config.num_epochs} | "
                f"Train: {train_losses['total_loss']:.4f} | "
                f"Val: {val_losses['total_loss']:.4f} | "
                f"Phys: {train_losses['physics_loss']:.4f} | "
                f"Drill: {train_losses['drill_loss']:.4f} | "
                f"CReg: {train_losses.get('correction_reg', 0):.4f} | "
                f"TF: {tf_ratio:.2f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.1e} | "
                f"Time: {epoch_time:.1f}s{best_marker}"
            )
            
            # Save checkpoint
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pth'
                self.save_checkpoint(checkpoint_path, is_best)
                if is_best:
                    self.logger.info(f"  💾 Saved best model (val_loss={self.best_val_loss:.4f})")
            
            # Plot learning curve after each epoch
            plot_learning_curve(self.history)
        
        self.logger.info("-" * 70)
        self.logger.info(f"Training complete! Best val loss: {self.best_val_loss:.4f}")


def load_data(config: TrainingConfig, logger: logging.Logger) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and split dataset from multiple checkpoint files."""
    # Find all checkpoint files
    checkpoint_files = sorted(glob.glob(config.data_pattern))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No files matching pattern: {config.data_pattern}")
    
    logger.info(f"Found {len(checkpoint_files)} data files")
    
    # Load and concatenate all checkpoint data
    all_T_maps = []
    all_P_maps = []
    all_h_values = []
    all_T_amb_values = []
    
    for f in checkpoint_files:
        data = np.load(f)
        all_T_maps.append(data['T_maps'])
        all_P_maps.append(data['P_maps'])
        all_h_values.append(data['h_values'])
        all_T_amb_values.append(data['T_amb_values'])
        logger.info(f"  Loaded {f}: {data['T_maps'].shape[0]} sequences")
    
    # Use only the largest checkpoint (contains all data)
    # The checkpoints are cumulative, so we use only the last/largest one
    largest_idx = np.argmax([t.shape[0] for t in all_T_maps])
    
    combined_data = {
        'T_maps': all_T_maps[largest_idx],
        'P_maps': all_P_maps[largest_idx],
        'h_values': all_h_values[largest_idx],
        'T_amb_values': all_T_amb_values[largest_idx],
    }
    
    n_sequences = combined_data['T_maps'].shape[0]
    logger.info(f"Using {n_sequences} total sequences from {checkpoint_files[largest_idx]}")
    
    # Deterministic, seeded split — reproducible across train/eval scripts.
    # (Historical note: before this fix the permutation was unseeded, so a
    # benchmark run after training re-split the data differently and its
    # "test" set overlapped the training set.)
    indices = np.random.default_rng(config.split_seed).permutation(n_sequences)
    logger.info(f"Train/val/test split seed: {config.split_seed}")
    
    # Split indices
    n_train = int(n_sequences * config.train_split)
    n_val = int(n_sequences * config.val_split)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = ThermalSequenceDataset(
        combined_data, train_idx, config, config.sequence_length
    )
    val_dataset = ThermalSequenceDataset(
        combined_data, val_idx, config, config.sequence_length
    )
    test_dataset = ThermalSequenceDataset(
        combined_data, test_idx, config, config.sequence_length
    )
    
    # Create dataloaders - M2 optimized (no pin_memory for MPS)
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': False,  # MPS doesn't support pin_memory
        'persistent_workers': config.persistent_workers and config.num_workers > 0,
        'prefetch_factor': config.prefetch_factor if config.num_workers > 0 else None,
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    logger.info(f"Train sequences: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    return train_loader, val_loader, test_loader


def main(resume_from: str = None):
    """Main training entry point."""
    # Setup logging
    logger = setup_logging('training.log')
    
    logger.info("=" * 70)
    logger.info("TRANSIENT THERMAL NEURAL OPERATOR - TRAINING")
    logger.info("=" * 70)
    
    # Configuration
    config = TrainingConfig()
    
    # M2 MPS optimizations
    if config.device == 'mps':
        # Enable MPS fallback for unsupported ops (prevents crashes)
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        logger.info("🍎 Apple Silicon M2 detected - MPS backend enabled")
    
    # Check for resume - prefer latest epoch checkpoint over best_model
    if resume_from == '__none__':
        resume_from = None
        logger.info("Starting fresh training (--fresh flag)")
    elif resume_from is None:
        epoch_checkpoints = list(Path(config.checkpoint_dir).glob('checkpoint_epoch_*.pth'))
        if epoch_checkpoints:
            def get_epoch_num(p):
                return int(p.stem.split('_')[-1])
            epoch_checkpoints.sort(key=get_epoch_num)
            resume_from = str(epoch_checkpoints[-1])
            logger.info(f"Found latest checkpoint: {resume_from}")
        else:
            best_path = Path(config.checkpoint_dir) / 'best_model.pth'
            if best_path.exists():
                resume_from = str(best_path)
                logger.info(f"Found best model checkpoint: {resume_from}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(config, logger)
    
    # Create model
    model = create_model(verify_params=True)
    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info(f"Device: {config.device}")
    logger.info("=" * 70)
    
    # Create trainer
    trainer = ThermalTrainer(model, config, logger)
    
    # Resume from checkpoint if available
    if resume_from and Path(resume_from).exists():
        trainer.load_checkpoint(resume_from)
        logger.info(f"▶️ Resuming from epoch {trainer.current_epoch + 1}")
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    logger.info("=" * 70)
    logger.info("Evaluating on test set...")
    test_losses = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_losses['total_loss']:.4f}")
    logger.info("=" * 70)


if __name__ == '__main__':
    import multiprocessing
    import argparse
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true', help='Train from scratch, ignore checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    if args.fresh:
        main(resume_from='__none__')
    else:
        main(resume_from=args.resume)
