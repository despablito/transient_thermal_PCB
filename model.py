"""
Physics-Augmented U-Net for Transient Thermal Prediction
=========================================================
Hybrid architecture: Implicit Euler physics baseline + Neural correction (U-Net).

    ΔT = ΔT_physics + ΔT_correction

- 48k parameters (< 147k STM32 budget)
- CoordConv encoder, Attention-Gated decoder, Depthwise Separable convolutions
- Zero-initialized correction → starts from pure physics baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CoordConv2d(nn.Module):
    """
    CoordConv layer that adds normalized X, Y coordinate channels.
    
    Helps the network understand spatial relationships in the thermal field.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = True):
        super().__init__()
        # +2 for X and Y coordinate channels
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size,
                              padding=padding, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        
        # Generate normalized coordinate grids [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        x_coords = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Expand to batch dimension
        xx = xx.unsqueeze(0).unsqueeze(0).expand(batch, 1, h, w)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(batch, 1, h, w)
        
        # Concatenate coordinates with input
        x_with_coords = torch.cat([x, xx, yy], dim=1)
        
        return self.conv(x_with_coords)


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution for parameter efficiency.
    
    Reduces parameters from (in_ch * out_ch * k²) to (in_ch * k² + in_ch * out_ch)
    Critical for STM32 deployment with <147k parameter budget.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    
    Focuses on thermal hotspots by learning to weight spatial features
    based on the gating signal from deeper layers.
    
    Reference: Attention U-Net (Oktay et al., 2018)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: Number of channels in gating signal
            F_l: Number of channels in skip connection
            F_int: Number of intermediate channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from coarser scale [B, F_g, H, W]
            x: Skip connection from encoder [B, F_l, H, W]
        
        Returns:
            Attention-weighted skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Handle size mismatch by interpolating gating signal
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class EncoderBlock(nn.Module):
    """Encoder block with separable convolutions and batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, use_coord: bool = False):
        super().__init__()
        
        if use_coord:
            self.conv1 = CoordConv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns pooled output and skip connection."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x), x


class DecoderBlock(nn.Module):
    """Decoder block with attention-gated skip connections."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.attention = AttentionGate(F_g=in_channels // 2, F_l=skip_channels,
                                       F_int=skip_channels // 2)
        
        combined_channels = in_channels // 2 + skip_channels
        self.conv1 = DepthwiseSeparableConv2d(combined_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply attention gate to skip connection
        skip = self.attention(x, skip)
        
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class Bottleneck(nn.Module):
    """Bottleneck with separable convolutions for parameter efficiency."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = DepthwiseSeparableConv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class RollingWindowThermalUNet(nn.Module):
    """
    Physics-Augmented Rolling Window U-Net for Transient Thermal Prediction.
    
    Hybrid architecture: Physics baseline (implicit Euler) + Neural correction.
    
    The physics residual provides an unconditionally stable 0D thermal estimate:
        ΔT_physics = (α·P - β·h·(T-T_amb)) / (1 + β·h)     [implicit Euler]
    The U-Net learns the spatial correction (diffusion, edge effects):
        ΔT = ΔT_physics + ΔT_correction
    
    This ensures:
    - Correct cooling behavior (ΔT < 0 when P=0, T > T_amb) from physics
    - Unconditional autoregressive stability from implicit formulation
    - Neural net only learns small corrections → faster convergence
    
    Input channels: P_t, P_{t-1}, T_{t-1}, T_amb, h (5 ch + CoordConv)
    Output: ΔT temperature rise map (1 ch)
    Target: <147k parameters for STM32 deployment.
    """
    
    ENCODER_CHANNELS = [16, 32, 48]
    BOTTLENECK_CHANNELS = 64
    DECODER_CHANNELS = [48, 32, 16]
    
    # Physics constants for implicit Euler residual (all in NORMALIZED space)
    # Derived from: dt=0.1s, P_scale=1e10, T_scale=300, h_scale=20000,
    #               ρCp=3210×690=2214900, thickness=0.5e-3
    POWER_COEFF = 1.5048    # dt × P_scale / (ρCp × T_scale)
    COOLING_COEFF = 1.8060  # dt × h_scale / (thickness × ρCp)
    
    def __init__(self, in_channels: int = 5, out_channels: int = 1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder path
        self.enc1 = EncoderBlock(in_channels, self.ENCODER_CHANNELS[0], use_coord=True)
        self.enc2 = EncoderBlock(self.ENCODER_CHANNELS[0], self.ENCODER_CHANNELS[1])
        self.enc3 = EncoderBlock(self.ENCODER_CHANNELS[1], self.ENCODER_CHANNELS[2])
        
        # Bottleneck
        self.bottleneck = Bottleneck(self.ENCODER_CHANNELS[2], self.BOTTLENECK_CHANNELS)
        
        # Decoder path
        self.dec3 = DecoderBlock(self.BOTTLENECK_CHANNELS, self.ENCODER_CHANNELS[2],
                                 self.DECODER_CHANNELS[0])
        self.dec2 = DecoderBlock(self.DECODER_CHANNELS[0], self.ENCODER_CHANNELS[1],
                                 self.DECODER_CHANNELS[1])
        self.dec1 = DecoderBlock(self.DECODER_CHANNELS[1], self.ENCODER_CHANNELS[0],
                                 self.DECODER_CHANNELS[2])
        
        # Output layer (learns correction to physics baseline)
        self.output_conv = nn.Conv2d(self.DECODER_CHANNELS[2], out_channels, 1)
        
        # Max correction magnitude in normalized space (0.1 ≈ ±30°C)
        self.CORRECTION_SCALE = 0.1
        
        self._init_weights()
        
        # Zero-init output layer: model starts from pure physics baseline
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def physics_residual(self, P_t: torch.Tensor, T_t_prev: torch.Tensor,
                         T_amb: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Implicit Euler 0D thermal model (unconditionally stable).
        
        ΔT = (α·P - β·h·(T_prev - T_amb)) / (1 + β·h)
        
        Stability: denominator (1 + β·h) > 1 always → |gain| < 1 for any h.
        """
        numerator = self.POWER_COEFF * P_t - self.COOLING_COEFF * h * (T_t_prev - T_amb)
        denominator = 1.0 + self.COOLING_COEFF * h
        return numerator / denominator
    
    def forward(self, P_t: torch.Tensor, P_t_prev: torch.Tensor,
                T_t_prev: torch.Tensor, T_amb: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        """
        Hybrid forward pass: physics baseline + neural correction.
        
        All inputs are NORMALIZED by their respective scales.
        Returns ΔT in normalized space.
        """
        # Physics baseline (0D implicit Euler — handles bulk thermodynamics)
        delta_T_physics = self.physics_residual(P_t, T_t_prev, T_amb, h)
        
        # Neural correction (U-Net — learns diffusion and spatial effects)
        x = torch.cat([P_t, P_t_prev, T_t_prev, T_amb, h], dim=1)
        
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        
        x = self.bottleneck(x3)
        
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        delta_T_correction = self.CORRECTION_SCALE * torch.tanh(self.output_conv(x))
        
        return delta_T_physics + delta_T_correction
    
    def predict_sequence(self, P_sequence: torch.Tensor,
                         T_initial: torch.Tensor,
                         T_amb: torch.Tensor,
                         h: torch.Tensor,
                         teacher_forcing_ratio: float = 0.0) -> torch.Tensor:
        """
        Predict temperature sequence autoregressively.
        
        Args:
            P_sequence: Power maps [B, T, 1, H, W]
            T_initial: Initial temperature [B, 1, H, W]
            T_amb: Ambient temperature [B, 1, H, W]
            h: Convection coefficient [B, 1, H, W]
            teacher_forcing_ratio: Probability of using ground truth T_{t-1}
        
        Returns:
            T_sequence: Predicted temperature maps [B, T, 1, H, W]
        """
        batch_size, seq_len = P_sequence.shape[:2]
        device = P_sequence.device
        
        T_predictions = []
        T_prev = T_initial
        P_prev = torch.zeros_like(P_sequence[:, 0])  # Zero power at t=-1
        
        for t in range(seq_len):
            P_t = P_sequence[:, t]
            
            # Predict ΔT
            delta_T = self.forward(P_t, P_prev, T_prev, T_amb, h)
            
            # Compute T_t = T_{t-1} + ΔT
            T_t = T_prev + delta_T
            T_predictions.append(T_t)
            
            # Update for next step
            P_prev = P_t
            T_prev = T_t
        
        return torch.stack(T_predictions, dim=1)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(verify_params: bool = True) -> RollingWindowThermalUNet:
    """
    Create and verify the thermal U-Net model.
    
    Args:
        verify_params: Whether to verify parameter count
    
    Returns:
        Initialized model
    """
    model = RollingWindowThermalUNet(in_channels=5, out_channels=1)
    
    n_params = model.count_parameters()
    print(f"Model Parameters: {n_params:,}")
    
    if verify_params and n_params > 147000:
        raise ValueError(f"Model has {n_params:,} parameters, exceeds 147k limit!")
    
    return model


def test_model():
    """Test model with dummy data."""
    model = create_model()
    
    # Create dummy inputs
    batch_size = 4
    H, W = 32, 32
    
    P_t = torch.randn(batch_size, 1, H, W)
    P_t_prev = torch.randn(batch_size, 1, H, W)
    T_t_prev = torch.randn(batch_size, 1, H, W) * 50 + 50  # 50-100°C
    T_amb = torch.full((batch_size, 1, H, W), 25.0)
    h = torch.full((batch_size, 1, H, W), 500.0)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        delta_T = model(P_t, P_t_prev, T_t_prev, T_amb, h)
    
    print(f"Input shapes:")
    print(f"  P_t: {P_t.shape}")
    print(f"  P_t_prev: {P_t_prev.shape}")
    print(f"  T_t_prev: {T_t_prev.shape}")
    print(f"  T_amb: {T_amb.shape}")
    print(f"  h: {h.shape}")
    print(f"Output shape: {delta_T.shape}")
    print(f"Output range: [{delta_T.min():.4f}, {delta_T.max():.4f}]")
    
    # Test sequence prediction
    seq_len = 10
    P_sequence = torch.randn(batch_size, seq_len, 1, H, W)
    T_initial = torch.full((batch_size, 1, H, W), 25.0)
    
    with torch.no_grad():
        T_sequence = model.predict_sequence(P_sequence, T_initial, T_amb, h)
    
    print(f"\nSequence prediction:")
    print(f"  P_sequence: {P_sequence.shape}")
    print(f"  T_sequence: {T_sequence.shape}")
    
    return model


if __name__ == '__main__':
    test_model()

