"""
Physics-Informed Validation Suite
=================================
Comprehensive stress tests for the Transient Thermal Neural Operator.

Tests:
1. Step Response (Heating dynamics with proper volumetric power)
2. Zero-Power Cooling (Newton's Law of Cooling)
3. Heat-then-Cool (Step up + step down)
4. Long-Horizon Stability (Sinusoidal power drive cycle)

All inputs use CORRECT normalization and spatial patterns matching training data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from model import create_model
from train import TrainingConfig
import logging

CHECKPOINT = "checkpoints/best_model.pth"
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
CFG = TrainingConfig()

# Physical constants matching data_gen.py
HOTSPOT_SIGMA = 2.5e-3   # [m] Gaussian hotspot size
THICKNESS = 0.5e-3        # [m] die thickness
LX, LY = 10e-3, 10e-3    # [m] domain size


def gaussian_pattern(peak_val: float, H=32, W=32, device=DEVICE):
    """Create Gaussian hotspot pattern matching training data spatial distribution."""
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sigma_norm = 2 * (HOTSPOT_SIGMA / LX) ** 2
    gauss = torch.exp(-(xx**2 + yy**2) / (2 * sigma_norm * (H/2)**2))
    gauss = torch.exp(-(xx**2 + yy**2) / 0.5)
    return (peak_val * gauss).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


def watts_to_volumetric(P_watts: float) -> float:
    """Convert total power [W] to peak volumetric density [W/m³] for Gaussian hotspot."""
    area = np.pi * HOTSPOT_SIGMA ** 2
    volume = area * THICKNESS
    return P_watts / volume


def run_simulation(model, P_watts_seq, T_initial_map, h_val, Tamb_val, steps):
    """
    Autoregressive simulation with CORRECT normalization and Gaussian patterns.
    
    Args:
        P_watts_seq: Power in WATTS (scalar or array) — converted to volumetric density
        T_initial_map: Initial temperature map [1,1,H,W] in °C (already spatial)
        h_val: Convection coefficient [W/(m²K)] — MUST be in training range [5000, 30000]
        Tamb_val: Ambient temperature [°C]
        steps: Number of simulation steps
    """
    H, W = 32, 32
    results_max = []
    results_mean = []
    results_center = []
    
    h_map = torch.full((1, 1, H, W), h_val, device=DEVICE) / CFG.h_scale
    Tamb_map = torch.full((1, 1, H, W), Tamb_val, device=DEVICE) / CFG.T_scale
    
    T_prev = T_initial_map.to(DEVICE) / CFG.T_scale
    P_prev = torch.zeros((1, 1, H, W), device=DEVICE)
    
    model.eval()
    with torch.no_grad():
        for t in range(steps):
            if isinstance(P_watts_seq, (float, int)):
                p_watts = P_watts_seq
            else:
                p_watts = P_watts_seq[t] if t < len(P_watts_seq) else P_watts_seq[-1]
            
            if p_watts > 0:
                q_vol = watts_to_volumetric(p_watts)
                P_t = gaussian_pattern(q_vol, H, W) / CFG.P_scale
            else:
                P_t = torch.zeros((1, 1, H, W), device=DEVICE)
            
            delta_T = model(P_t, P_prev, T_prev, Tamb_map, h_map)
            T_pred = T_prev + delta_T
            
            T_denorm = T_pred * CFG.T_scale
            results_max.append(T_denorm.max().item())
            results_mean.append(T_denorm.mean().item())
            results_center.append(T_denorm[0, 0, H//2, W//2].item())
            
            T_prev = T_pred
            P_prev = P_t
    
    return {
        'max': np.array(results_max),
        'mean': np.array(results_mean),
        'center': np.array(results_center),
    }


def validate_physics(checkpoint=CHECKPOINT, fresh=False):
    print(f"Device: {DEVICE}")
    print(f"Normalization: P_scale={CFG.P_scale:.0e}, T_scale={CFG.T_scale}, h_scale={CFG.h_scale}")
    
    model = create_model(verify_params=False).to(DEVICE)
    
    if fresh:
        epoch = 'fresh (physics-only)'
        print(f"Using FRESH model (no training, physics residual only)")
    else:
        print(f"Loading model from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f"Model loaded (epoch {epoch})")
    
    H, W = 32, 32
    h_val = 15000.0   # Middle of training range [5000, 30000]
    Tamb = 35.0        # Typical ambient
    dt = 0.1           # Time step [s]
    sim_steps = 100
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    scores = {}
    
    # =========================================================================
    # TEST 1: STEP RESPONSE (Heating)
    # =========================================================================
    print("\n[Test 1] Step Response: 0 -> 80W constant power...")
    
    T_init_uniform = torch.full((1, 1, H, W), Tamb, device=DEVICE)
    P_step = 80.0  # 80W — mid-range for SiC
    
    res_step = run_simulation(model, P_step, T_init_uniform, h_val, Tamb, sim_steps)
    
    ax1 = fig.add_subplot(gs[0, 0])
    t_axis = np.arange(sim_steps) * dt
    ax1.plot(t_axis, res_step['center'], 'r-', linewidth=2, label='Center (hotspot)')
    ax1.plot(t_axis, res_step['mean'], 'b--', linewidth=1.5, label='Mean')
    ax1.axhline(y=Tamb, color='k', linestyle=':', alpha=0.5, label=f'T_amb={Tamb}°C')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Temperature [°C]')
    ax1.set_title(f'Test 1: Step Response (P={P_step}W, h={h_val:.0f})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    monotonic = np.all(np.diff(res_step['center']) >= -0.5)
    above_amb = np.all(np.array(res_step['center']) >= Tamb - 1.0)
    t1_pass = monotonic and above_amb
    scores['step_response'] = t1_pass
    
    status = "PASS" if t1_pass else "FAIL"
    color = 'green' if t1_pass else 'red'
    ax1.text(0.02, 0.95, f"{status}: {'Monotonic rise' if monotonic else 'Oscillations'}", 
             transform=ax1.transAxes, color=color, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"  Center: {res_step['center'][0]:.1f} -> {res_step['center'][-1]:.1f}°C")
    print(f"  Monotonic: {monotonic} | Above ambient: {above_amb}")
    
    # =========================================================================
    # TEST 2: COOLING (Newton's Law)
    # =========================================================================
    print("\n[Test 2] Cooling: Hotspot at 200°C, P=0, should cool to T_amb...")
    
    T_hot = gaussian_pattern(200.0 - Tamb, H, W) + Tamb
    T_init_hot = T_hot.clone()
    
    res_cool = run_simulation(model, 0.0, T_init_hot, h_val, Tamb, sim_steps)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_axis, res_cool['center'], 'b-', linewidth=2, label='Center (hotspot)')
    ax2.plot(t_axis, res_cool['mean'], 'c--', linewidth=1.5, label='Mean')
    ax2.axhline(y=Tamb, color='g', linestyle='--', alpha=0.7, label=f'T_amb={Tamb}°C')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Temperature [°C]')
    ax2.set_title(f'Test 2: Cooling Test (T₀=200°C, P=0W, h={h_val:.0f})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    cooling_ok = np.all(np.diff(res_cool['center']) <= 0.5)
    above_ambient = np.min(res_cool['center']) >= Tamb - 2.0
    final_decrease = res_cool['center'][-1] < res_cool['center'][0]
    t2_pass = cooling_ok and above_ambient and final_decrease
    scores['cooling'] = t2_pass
    
    status = "PASS" if t2_pass else "FAIL"
    color = 'green' if t2_pass else 'red'
    detail = f"Δ={res_cool['center'][-1]-res_cool['center'][0]:.1f}°C"
    if not above_ambient:
        detail = f"Below ambient: min={np.min(res_cool['center']):.1f}°C"
    if not cooling_ok:
        detail = "Oscillations detected"
    ax2.text(0.02, 0.95, f"{status}: {detail}",
             transform=ax2.transAxes, color=color, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"  Center: {res_cool['center'][0]:.1f} -> {res_cool['center'][-1]:.1f}°C")
    print(f"  Monotonic cooling: {cooling_ok} | Above ambient: {above_ambient}")
    
    # =========================================================================
    # TEST 3: HEAT THEN COOL
    # =========================================================================
    print("\n[Test 3] Heat+Cool: 50 steps ON (60W), then 50 steps OFF...")
    
    T_init_amb = torch.full((1, 1, H, W), Tamb, device=DEVICE)
    half = sim_steps // 2
    P_on_off = np.zeros(sim_steps)
    P_on_off[:half] = 60.0
    
    res_hc = run_simulation(model, P_on_off, T_init_amb, h_val, Tamb, sim_steps)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_axis, res_hc['center'], 'r-', linewidth=2, label='Center T')
    ax3.axvline(x=half * dt, color='k', linestyle=':', alpha=0.5, label='Power OFF')
    ax3.axhline(y=Tamb, color='g', linestyle='--', alpha=0.5)
    
    ax3_pw = ax3.twinx()
    ax3_pw.fill_between(t_axis, P_on_off, alpha=0.15, color='orange', label='Power [W]')
    ax3_pw.set_ylabel('Power [W]', color='orange')
    ax3_pw.set_ylim(0, 120)
    ax3_pw.tick_params(axis='y', labelcolor='orange')
    
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Temperature [°C]')
    ax3.set_title(f'Test 3: Heat→Cool (60W on/off, h={h_val:.0f})')
    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    heating_phase = np.diff(res_hc['center'][:half])
    cooling_phase = np.diff(res_hc['center'][half:])
    heats_up = np.mean(heating_phase) > 0
    cools_down = np.mean(cooling_phase) < 0
    peak_at_transition = np.argmax(res_hc['center']) >= half - 5
    t3_pass = heats_up and cools_down
    scores['heat_cool'] = t3_pass
    
    status = "PASS" if t3_pass else "FAIL"
    color = 'green' if t3_pass else 'red'
    ax3.text(0.02, 0.95, f"{status}: Heat={'OK' if heats_up else 'FAIL'}, Cool={'OK' if cools_down else 'FAIL'}",
             transform=ax3.transAxes, color=color, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"  Peak at step ~{np.argmax(res_hc['center'])}: {max(res_hc['center']):.1f}°C")
    print(f"  Heats up: {heats_up} | Cools down: {cools_down}")
    
    # =========================================================================
    # TEST 4: LONG HORIZON STABILITY
    # =========================================================================
    print("\n[Test 4] Long Horizon: 300 steps with sinusoidal power...")
    
    long_steps = 300
    t_long = np.arange(long_steps) * dt
    P_sine = 50.0 * (np.sin(2 * np.pi * t_long / 5.0) + 1)
    
    T_init_warm = torch.full((1, 1, H, W), Tamb + 10.0, device=DEVICE)
    
    res_long = run_simulation(model, P_sine, T_init_warm, h_val, Tamb, long_steps)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t_long, res_long['center'], 'k-', linewidth=1.5, label='Center T')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Temperature [°C]')
    
    ax4_pw = ax4.twinx()
    ax4_pw.plot(t_long, P_sine, 'orange', alpha=0.4, linewidth=1, label='Power [W]')
    ax4_pw.set_ylabel('Power [W]', color='orange')
    ax4_pw.tick_params(axis='y', labelcolor='orange')
    
    ax4.set_title(f'Test 4: Long Horizon ({long_steps} steps, sinusoidal)')
    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    exploded = np.max(np.abs(res_long['center'])) > 500
    below_zero = np.min(res_long['center']) < 0
    t4_pass = not exploded and not below_zero
    scores['long_horizon'] = t4_pass
    
    status = "PASS" if t4_pass else "FAIL"
    color = 'green' if t4_pass else 'red'
    detail = "Stable"
    if exploded:
        detail = f"Explosion: max={np.max(res_long['center']):.0f}°C"
    elif below_zero:
        detail = f"Negative T: min={np.min(res_long['center']):.1f}°C"
    ax4.text(0.02, 0.95, f"{status}: {detail}",
             transform=ax4.transAxes, color=color, fontweight='bold', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"  Range: [{np.min(res_long['center']):.1f}, {np.max(res_long['center']):.1f}]°C")
    
    # =========================================================================
    # OVERALL SCORE
    # =========================================================================
    n_pass = sum(scores.values())
    n_total = len(scores)
    
    fig.suptitle(
        f'Physics Validation Report — {n_pass}/{n_total} Tests Passed '
        f'(Epoch {epoch}, h={h_val:.0f} W/(m²K))',
        fontsize=14, fontweight='bold',
        color='green' if n_pass == n_total else 'red'
    )
    
    plt.savefig('visualizations/physics_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{n_total} tests passed")
    for name, passed in scores.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print(f"{'='*60}")
    print("Saved: visualizations/physics_validation.png")
    
    return scores


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Path to model checkpoint')
    parser.add_argument('--fresh', action='store_true', help='Use fresh (untrained) model')
    args = parser.parse_args()
    validate_physics(checkpoint=args.checkpoint, fresh=args.fresh)
