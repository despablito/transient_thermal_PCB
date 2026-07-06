"""
Transient Thermal Data Generator
=================================
2D FDM solver + electro-thermal SiC MOSFET loss model.

Generates training data: 1000 sequences × 100 frames × 32×32 temperature
and power density maps for the thermal surrogate model.

Physics:
    ρCp ∂T/∂t = k∇²T + Q(t) − h(T − T_amb)/L

KNOWN MODELING CONVENTIONS (do not change without retraining the model —
the network was trained on data produced with exactly these conventions):

1. Power normalization: `calculate_volumetric_power_density` divides by the
   Gaussian's π·σ² "disk" area, but the actual 2D Gaussian integrates to
   2π·σ². Net effect: the power injected into the domain is ≈1.85× the
   nominal P_total from the loss model (2× minus boundary truncation).
   The dataset is internally self-consistent; only the absolute wattage
   labels are nominal, not effective.

2. Temporal aliasing: the electrical sine (f_out = 10-100 Hz) is sampled at
   dt = 0.1 s (Nyquist 5 Hz), so the per-frame loss is an aliased sample of
   the instantaneous loss curve, not its period average. For f_out near
   multiples of 5 Hz the sampled power of a segment can be biased anywhere
   between 0 and 2× the true average. The excitation is therefore best read
   as a *randomized power profile*, not a literal drive cycle. The physically
   correct approach (period-averaged losses: P_cond = I_peak²·R/2,
   P_sw ∝ 2·I_peak/π) requires regenerating data and retraining.

3. One-way electro-thermal coupling: R_ds(on) is a constant R_ds_eff; the
   temperature feedback T → R_ds(on) (temp_coeff) is NOT applied.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_file: str = 'generator.log') -> logging.Logger:
    logger = logging.getLogger('ThermalGenerator')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'))
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

T_VIS_MIN = 20.0
T_VIS_MAX = 250.0

# =============================================================================
# SiC MOSFET ELECTRO-THERMAL LOSS MODEL
# =============================================================================

@dataclass
class SiCLossModel:
    """
    Electro-Thermal Loss Model for 1200V SiC MOSFET.
    
    Calculates conduction and switching losses based on electrical operating point.
    
    Physics:
        P_cond = I(t)² × R_ds_eff
        P_sw = f_sw × V_dc × |I(t)| × E_sw_scale
        P_total = P_cond + P_sw
    """
    # Electrical parameters (randomized per sequence)
    V_dc: float = 600.0           # DC Bus Voltage [V] (400-800V range)
    I_peak: float = 100.0         # Peak Load Current [A] (20-75A range)
    f_sw: float = 20e3            # Switching Frequency [Hz] - constant 20kHz
    f_out: float = 50.0           # Motor/Output Frequency [Hz] (10-100Hz range)
    
    # SiC MOSFET Parameters (Typical 1200V/100A die)
    # NOTE: coupling is one-way (electrical → thermal). R_ds_on_base and
    # temp_coeff are declared for reference but NOT used — the simulation
    # uses a constant R_ds_eff (~hot-die value), with no T → R_ds feedback.
    R_ds_on_base: float = 0.020   # Base on-resistance [Ohm] at 25°C (unused)
    temp_coeff: float = 0.005     # Temperature coefficient [1/°C] (unused)
    R_ds_eff: float = 0.030       # Effective R_ds(on) for simulation [Ohm]

    # Empirical switching loss factor: P_sw = f_sw × V_dc × |I| × E_sw_scale.
    # 1e-8 J/(A·V) = 0.01 µJ per A·V. This is deliberately LOW compared to
    # datasheet E_on+E_off of a 1200V SiC die (order 5-20 µJ/(A·V) at rated
    # conditions) — it keeps switching losses a minor contribution vs
    # conduction losses in this dataset.
    E_sw_scale: float = 1.0e-8
    
    def current_at_time(self, t: float) -> float:
        """
        Calculate instantaneous load current I(t).
        
        I(t) = I_peak × sin(2π × f_out × t)
        """
        return self.I_peak * np.sin(2 * np.pi * self.f_out * t)
    
    def conduction_loss(self, I: float) -> float:
        """
        Calculate conduction losses.
        
        P_cond = I² × R_ds_eff
        """
        return I**2 * self.R_ds_eff
    
    def switching_loss(self, I: float) -> float:
        """
        Calculate switching losses (averaged over PWM cycle).
        
        P_sw = f_sw × V_dc × |I| × E_sw_scale
        """
        return self.f_sw * self.V_dc * abs(I) * self.E_sw_scale
    
    def total_loss_at_time(self, t: float) -> float:
        """
        Calculate total power loss at time t.
        
        P_total = P_cond + P_sw
        """
        I = self.current_at_time(t)
        P_cond = self.conduction_loss(I)
        P_sw = self.switching_loss(I)
        return P_cond + P_sw
    
    def get_loss_profile(self, t_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power loss and current profiles over time array.
        
        Returns:
            P_loss: Total power loss [W] at each time point
            I_load: Load current [A] at each time point
        """
        I_load = self.I_peak * np.sin(2 * np.pi * self.f_out * t_array)
        P_cond = I_load**2 * self.R_ds_eff
        P_sw = self.f_sw * self.V_dc * np.abs(I_load) * self.E_sw_scale
        P_loss = P_cond + P_sw
        return P_loss, I_load


# =============================================================================
# THERMAL PROPERTIES
# =============================================================================

@dataclass
class ThermalProperties:
    """Material and simulation properties for SiC MOSFET thermal model."""
    # SiC thermal properties
    k: float = 120.0          # Thermal conductivity [W/(m·K)]
    rho: float = 3210.0       # Density [kg/m³]
    Cp: float = 690.0         # Specific heat capacity [J/(kg·K)]
    
    # Geometry
    Lx: float = 10e-3         # 10mm
    Ly: float = 10e-3         # 10mm
    thickness: float = 0.5e-3 # 0.5mm
    
    # Grid
    nx: int = 32
    ny: int = 32
    
    # Boundary conditions
    T_amb_range: Tuple[float, float] = (25.0, 60.0)
    
    # Convection (h) range: Liquid/High Performance Heatsink
    h_range: Tuple[float, float] = (5000.0, 30000.0)
    
    # Electrical operating ranges (for SiCLossModel)
    V_dc_range: Tuple[float, float] = (400.0, 800.0)
    I_peak_range: Tuple[float, float] = (20.0, 75.0)
    f_out_range: Tuple[float, float] = (10.0, 100.0)
    f_sw: float = 20e3  # Fixed 20 kHz
    
    # Time parameters
    dt: float = 0.1           
    n_frames: int = 100       
    
    @property
    def dx(self) -> float: return self.Lx / (self.nx - 1)
    @property
    def dy(self) -> float: return self.Ly / (self.ny - 1)
    @property
    def thermal_diffusivity(self) -> float: return self.k / (self.rho * self.Cp)
    @property
    def heat_capacity_volumetric(self) -> float: return self.rho * self.Cp
    @property
    def domain_volume(self) -> float: return self.Lx * self.Ly * self.thickness


# =============================================================================
# TRANSIENT THERMAL SOLVER (UNCHANGED)
# =============================================================================

class TransientThermalSolver:
    """2D Transient FDM solver for heat equation."""
    
    def __init__(self, props: ThermalProperties):
        self.props = props
        self.nx = props.nx
        self.ny = props.ny
        self.rho_Cp = props.heat_capacity_volumetric
        
        # Laplacian kernel
        self.dx2 = props.dx ** 2
        self.dy2 = props.dy ** 2
        self.laplacian_kernel = np.array([
            [0, 1/self.dy2, 0],
            [1/self.dx2, -2/self.dx2 - 2/self.dy2, 1/self.dx2],
            [0, 1/self.dy2, 0]
        ])
        
    def compute_laplacian(self, T: np.ndarray) -> np.ndarray:
        return convolve(T, self.laplacian_kernel, mode='constant', cval=0.0)
    
    def heat_equation_rhs(self, t: float, T_flat: np.ndarray,
                          Q_func, h: float, T_amb: float) -> np.ndarray:
        T = T_flat.reshape(self.ny, self.nx)
        
        diffusion = self.props.k * self.compute_laplacian(T)
        Q = Q_func(t)
        
        # Effective cooling via bottom/top surface
        convection = -h * (T - T_amb) / self.props.thickness
        
        dTdt = (diffusion + Q + convection) / self.rho_Cp
        
        # Adiabatic edges
        dTdt[0, :] = dTdt[1, :]
        dTdt[-1, :] = dTdt[-2, :]
        dTdt[:, 0] = dTdt[:, 1]
        dTdt[:, -1] = dTdt[:, -2]
        
        return dTdt.flatten()
    
    def solve(self, Q_func, h: float, T_amb: float, t_span, t_eval):
        T_initial = np.full((self.ny, self.nx), T_amb)
        
        solution = solve_ivp(
            fun=lambda t, T: self.heat_equation_rhs(t, T, Q_func, h, T_amb),
            t_span=t_span,
            y0=T_initial.flatten(),
            method='BDF', 
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-4
        )
        return solution.t, solution.y.T.reshape(-1, self.ny, self.nx)


# =============================================================================
# POWER PROFILE GENERATOR (UPDATED: Electro-Thermal Model)
# =============================================================================

class PowerProfileGenerator:
    """
    Generate power profiles using SiC Electro-Thermal Loss Model.
    
    Features:
    - Sinusoidal current → Losses pulsate at 2×f_out
    - Drive Cycles: I_peak and f_out change every 2-3 seconds
    - Spatial Gaussian hotspot distribution
    """
    
    def __init__(self, props: ThermalProperties, seed: Optional[int] = None):
        self.props = props
        self.rng = np.random.default_rng(seed)
        
        # Coordinate grids for spatial distribution
        x = np.linspace(0, props.Lx, props.nx)
        y = np.linspace(0, props.Ly, props.ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Fixed hotspot location (die center - chip doesn't move)
        self.hotspot_cx = props.Lx / 2
        self.hotspot_cy = props.Ly / 2
        self.hotspot_sigma = 2.5e-3  # 2.5mm effective hotspot size
        
    def calculate_volumetric_power_density(self, P_watts: float, sigma: float) -> float:
        """
        Convert total power [W] to PEAK volumetric density [W/m³].

        CONVENTION (kept for compatibility with the trained model — see module
        docstring, point 1): normalizes by the π·σ² disk area, while the full
        Gaussian integrates to 2π·σ². The effective power injected into the
        domain is therefore ≈1.85× P_watts. Fixing this to 2π·σ² requires
        regenerating the dataset and retraining.
        """
        area_hotspot = np.pi * sigma ** 2
        volume_hotspot = area_hotspot * self.props.thickness
        return P_watts / volume_hotspot
    
    def gaussian_hotspot(self, q_volumetric: float) -> np.ndarray:
        """Generate Gaussian heat source distribution centered on die."""
        r2 = (self.X - self.hotspot_cx)**2 + (self.Y - self.hotspot_cy)**2
        return q_volumetric * np.exp(-r2 / (2 * self.hotspot_sigma**2))
    
    def generate_drive_cycle(self, n_frames: int, dt: float) -> List[dict]:
        """
        Generate drive cycle segments with varying I_peak and f_out.
        
        Simulates acceleration/braking/idle events in motor drive application.
        Each segment lasts 1-3 seconds.
        ~25% of segments are "idle" (I_peak=0) to provide cooling data.
        """
        total_time = n_frames * dt
        segments = []
        t = 0.0
        
        while t < total_time:
            duration = self.rng.uniform(1.0, 3.0)
            end_t = min(t + duration, total_time)
            
            # 25% chance of idle segment (motor off — cooling phase)
            if self.rng.random() < 0.25:
                I_peak = 0.0
                f_out = 0.0
                V_dc = 0.0
            else:
                V_dc = self.rng.uniform(*self.props.V_dc_range)
                I_peak = self.rng.uniform(*self.props.I_peak_range)
                f_out = self.rng.uniform(*self.props.f_out_range)
            
            segments.append({
                't_start': t,
                't_end': end_t,
                'V_dc': V_dc,
                'I_peak': I_peak,
                'f_out': f_out
            })
            
            t = end_t
        
        return segments
    
    def generate_power_sequence(self, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate power sequence using electro-thermal loss model.
        
        Returns:
            P_sequence: Volumetric power density maps [n_frames, ny, nx]
            I_sequence: Load current at each frame [n_frames]
        """
        dt = self.props.dt
        t_array = np.arange(n_frames) * dt
        
        P_sequence = np.zeros((n_frames, self.props.ny, self.props.nx))
        I_sequence = np.zeros(n_frames)
        P_total_sequence = np.zeros(n_frames)  # For logging
        
        # Generate drive cycle
        segments = self.generate_drive_cycle(n_frames, dt)
        
        for seg in segments:
            # Create loss model for this segment
            loss_model = SiCLossModel(
                V_dc=seg['V_dc'],
                I_peak=seg['I_peak'],
                f_sw=self.props.f_sw,
                f_out=seg['f_out']
            )
            
            # Find frame indices for this segment
            start_frame = int(seg['t_start'] / dt)
            end_frame = min(int(seg['t_end'] / dt), n_frames)
            
            for frame in range(start_frame, end_frame):
                t = frame * dt
                
                # Calculate current and losses
                I = loss_model.current_at_time(t)
                P_total = loss_model.total_loss_at_time(t)
                
                I_sequence[frame] = I
                P_total_sequence[frame] = P_total
                
                # Convert to volumetric density and apply spatial distribution
                q_vol = self.calculate_volumetric_power_density(P_total, self.hotspot_sigma)
                P_sequence[frame] = self.gaussian_hotspot(q_vol)
        
        return P_sequence, I_sequence

    def create_power_interpolator(self, P_sequence: np.ndarray, dt: float):
        """Create interpolation function Q(t) for the thermal solver."""
        n_frames = len(P_sequence)
        
        def Q_func(t: float) -> np.ndarray:
            idx = int(t / dt)
            idx = np.clip(idx, 0, n_frames - 1)
            
            # Linear interpolation
            if idx < n_frames - 1:
                alpha = (t / dt) - idx
                return (1 - alpha) * P_sequence[idx] + alpha * P_sequence[idx + 1]
            return P_sequence[idx]
        
        return Q_func


# =============================================================================
# GIF GENERATION (UPDATED: Current on secondary axis)
# =============================================================================

def generate_sequence_gif(T_maps, P_maps, I_load, h, T_amb, dt, output_path, fps=15):
    """
    Generate 3-Panel GIF with Current visualization.
    
    Left: Power Density Map
    Center: Temperature Map
    Right: Time History (Max Temp [°C] and Current [A])
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    n_frames = len(T_maps)
    
    # Calculate scales
    P_dens_max = P_maps.max() if P_maps.max() > 0 else 1.0
    T_data_max = T_maps.max()
    T_vmax = min(max(100.0, T_data_max * 1.1), 400.0)
    T_vmin = T_VIS_MIN
    
    I_max = max(abs(I_load).max() * 1.2, 10.0)
    
    # Time vector
    t_vec = np.arange(n_frames) * dt
    
    # Calculate max temps
    max_temps = T_maps.max(axis=(1, 2))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'SiC MOSFET Thermal Simulation | h={h:.0f} W/(m²K) | MaxT={T_data_max:.0f}°C', 
                 fontweight='bold')
    
    # 1. Power Map
    im_P = axes[0].imshow(P_maps[0], cmap='inferno', origin='lower', vmin=0, vmax=P_dens_max)
    axes[0].set_title('Power Density [W/m³]')
    axes[0].axis('off')
    plt.colorbar(im_P, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Temp Map
    im_T = axes[1].imshow(T_maps[0], cmap='turbo', origin='lower', vmin=T_vmin, vmax=T_vmax)
    axes[1].set_title('Junction Temperature [°C]')
    axes[1].axis('off')
    plt.colorbar(im_T, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Time History: Temperature and Current
    ax_temp = axes[2]
    ax_curr = ax_temp.twinx()
    
    # Ghost lines (full history, faded)
    ax_temp.plot(t_vec, max_temps, 'r-', alpha=0.2)
    ax_curr.plot(t_vec, I_load, 'b-', alpha=0.2)
    
    # Dynamic lines
    line_T, = ax_temp.plot([], [], 'r-', linewidth=2, label='Max T [°C]')
    line_I, = ax_curr.plot([], [], 'b-', linewidth=1.5, label='Current [A]')
    cursor_line = ax_temp.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    
    # Zero line for current
    ax_curr.axhline(y=0, color='b', linestyle='--', alpha=0.3)
    
    # Setup axes
    ax_temp.set_xlim(0, t_vec[-1])
    ax_temp.set_ylim(T_vmin, T_vmax)
    ax_curr.set_ylim(-I_max, I_max)
    
    ax_temp.set_xlabel('Time [s]')
    ax_temp.set_ylabel('Max Temp [°C]', color='r')
    ax_curr.set_ylabel('Current [A]', color='b')
    ax_temp.set_title('Transient Response')
    ax_temp.grid(True, alpha=0.3)
    
    ax_temp.tick_params(axis='y', labelcolor='r')
    ax_curr.tick_params(axis='y', labelcolor='b')
    
    # Legend
    lines = [line_T, line_I]
    labels = [l.get_label() for l in lines]
    ax_temp.legend(lines, labels, loc='upper left', fontsize='small')
    
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    def update(frame):
        im_P.set_array(P_maps[frame])
        im_T.set_array(T_maps[frame])
        
        current_t = t_vec[:frame+1]
        line_T.set_data(current_t, max_temps[:frame+1])
        line_I.set_data(current_t, I_load[:frame+1])
        cursor_line.set_xdata([t_vec[frame]])
        
        time_text.set_text(f't = {frame * dt:.2f} s | I = {I_load[frame]:.1f} A')
        return [im_P, im_T, line_T, line_I, cursor_line, time_text]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# =============================================================================
# MAIN DATASET GENERATION
# =============================================================================

def generate_dataset(n_sequences=1000, n_frames=100, output_path='transient_data.npz', 
                     seed=42, gif_interval=100, checkpoint_interval=100):
    """
    Generate transient thermal dataset using electro-thermal loss model.
    
    Args:
        checkpoint_interval: Save intermediate .npz every N sequences (0 = disabled)
    """
    logger = setup_logging()
    props = ThermalProperties(n_frames=n_frames)
    solver = TransientThermalSolver(props)
    
    logger.info("=" * 70)
    logger.info("SiC MOSFET ELECTRO-THERMAL DATA GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Cooling: h = {props.h_range} W/(m²K)")
    logger.info(f"V_dc: {props.V_dc_range} V | I_peak: {props.I_peak_range} A")
    logger.info(f"f_sw: {props.f_sw/1e3:.0f} kHz | f_out: {props.f_out_range} Hz")
    logger.info(f"T_amb: {props.T_amb_range} °C")
    logger.info("=" * 70)
    
    T_seqs, P_seqs, I_seqs = [], [], []
    h_vals, T_amb_vals, V_dc_vals, I_peak_vals = [], [], [], []
    
    rng = np.random.default_rng(seed)
    gif_dir = Path('visualizations')
    gif_dir.mkdir(exist_ok=True)
    
    for seq_idx in tqdm(range(n_sequences), desc="Simulating"):
        # Random operating conditions
        h = rng.uniform(*props.h_range)
        T_amb = rng.uniform(*props.T_amb_range)
        
        # Generate power profile with electro-thermal model
        p_gen = PowerProfileGenerator(props, seed=seed+seq_idx)
        P_seq, I_seq = p_gen.generate_power_sequence(n_frames)
        Q_func = p_gen.create_power_interpolator(P_seq, props.dt)
        
        # Solve thermal equation
        _, T_hist = solver.solve(
            Q_func, h, T_amb, 
            (0, props.dt * n_frames), 
            np.linspace(0, props.dt * n_frames, n_frames + 1)
        )
        
        T_res = T_hist[1:].astype(np.float32)
        P_res = P_seq.astype(np.float32)
        I_res = I_seq.astype(np.float32)
        
        T_max = T_res.max()
        I_max = np.abs(I_seq).max()
        
        # Log with electrical info
        log_msg = f"Seq {seq_idx:04d} | h={h:.0f} | I_max={I_max:.0f}A | T_max={T_max:.1f}°C"
        if T_max > 400.0:
            logger.warning(f"{log_msg} [OVERHEAT]")
        elif T_max < T_amb + 5.0:
            logger.warning(f"{log_msg} [NO HEAT]")
        else:
            logger.info(log_msg)
        
        # Store results
        T_seqs.append(T_res)
        P_seqs.append(P_res)
        I_seqs.append(I_res)
        h_vals.append(h)
        T_amb_vals.append(T_amb)
        
        # Generate GIF
        if gif_interval > 0 and seq_idx % gif_interval == 0:
            generate_sequence_gif(
                T_res, P_res, I_res, h, T_amb, props.dt,
                str(gif_dir / f'seq_{seq_idx:04d}.gif')
            )
        
        # Save checkpoint
        if checkpoint_interval > 0 and (seq_idx + 1) % checkpoint_interval == 0:
            checkpoint_path = f'checkpoint_{seq_idx+1:04d}.npz'
            checkpoint_data = {
                'T_maps': np.array(T_seqs),
                'P_maps': np.array(P_seqs),
                'I_load': np.array(I_seqs),
                'h_values': np.array(h_vals),
                'T_amb_values': np.array(T_amb_vals),
                'dt': props.dt, 'dx': props.dx, 'dy': props.dy, 'f_sw': props.f_sw,
            }
            np.savez_compressed(checkpoint_path, **checkpoint_data)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path} ({seq_idx+1} sequences)")
    
    # Save final dataset
    data = {
        'T_maps': np.array(T_seqs),
        'P_maps': np.array(P_seqs),
        'I_load': np.array(I_seqs),
        'h_values': np.array(h_vals),
        'T_amb_values': np.array(T_amb_vals),
        'dt': props.dt,
        'dx': props.dx,
        'dy': props.dy,
        'f_sw': props.f_sw,
    }
    
    np.savez_compressed(output_path, **data)
    logger.info("=" * 70)
    logger.info(f"Dataset saved: {output_path}")
    logger.info(f"T_maps: {data['T_maps'].shape}")
    logger.info(f"I_load: {data['I_load'].shape}")
    logger.info(f"Temp range: [{data['T_maps'].min():.1f}, {data['T_maps'].max():.1f}] °C")
    logger.info("=" * 70)
    
    return data


if __name__ == '__main__':
    generate_dataset(n_sequences=1000, gif_interval=50, checkpoint_interval=100)
