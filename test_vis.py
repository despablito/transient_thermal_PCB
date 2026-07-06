"""
Model Diagnostic Visualization
===============================
Generates portfolio-quality diagnostic plots comparing:
- Ground Truth (FDM solver)
- AI One-Step (teacher forcing — checks per-step accuracy)
- AI Autoregressive (full autonomous — checks stability)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from train import TrainingConfig, ThermalSequenceDataset, load_data
from model import create_model
import logging
import argparse

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def visualize_diagnostic(model, dataset, idx=0, save_path='visualizations/diagnostic_plot.png'):
    model.eval()
    data = dataset[idx]
    cfg = dataset.config
    
    T_seq = data['T_seq'].unsqueeze(0).to(DEVICE)   # [1, T, 1, H, W]
    P_seq = data['P_seq'].unsqueeze(0).to(DEVICE)
    T_initial = data['T_initial'].unsqueeze(0).to(DEVICE)
    P_initial = data['P_initial'].unsqueeze(0).to(DEVICE)
    h = data['h'].unsqueeze(0).to(DEVICE)
    T_amb = data['T_amb'].unsqueeze(0).to(DEVICE)
    
    H, W = 32, 32
    h_map = h.view(1, 1, 1, 1).expand(1, 1, H, W) / cfg.h_scale
    Tamb_map = T_amb.view(1, 1, 1, 1).expand(1, 1, H, W) / cfg.T_scale
    
    pred_autoreg = []
    pred_onestep = []
    ground_truth = []
    delta_T_pred_list = []
    delta_T_true_list = []
    
    # 1. Autoregressive Prediction (Hard Mode)
    T_prev = T_initial / cfg.T_scale
    P_prev = P_initial / cfg.P_scale
    
    with torch.no_grad():
        for t in range(T_seq.shape[1]):
            P_t = P_seq[:, t] / cfg.P_scale
            delta_T = model(P_t, P_prev, T_prev, Tamb_map, h_map)
            T_pred = T_prev + delta_T
            pred_autoreg.append(T_pred.cpu().numpy() * cfg.T_scale)
            T_prev = T_pred
            P_prev = P_t

    # 2. One-Step Prediction (Easy Mode — Teacher Forcing)
    T_prev = T_initial / cfg.T_scale
    P_prev = P_initial / cfg.P_scale
    
    with torch.no_grad():
        for t in range(T_seq.shape[1]):
            P_t = P_seq[:, t] / cfg.P_scale
            T_target = T_seq[:, t]
            
            delta_T = model(P_t, P_prev, T_prev, Tamb_map, h_map)
            T_pred = T_prev + delta_T
            
            pred_onestep.append(T_pred.cpu().numpy() * cfg.T_scale)
            ground_truth.append(T_target.cpu().numpy())
            
            delta_T_pred_list.append(delta_T.cpu().numpy() * cfg.T_scale)
            delta_T_true = (T_target - T_prev * cfg.T_scale).cpu().numpy()
            delta_T_true_list.append(delta_T_true)
            
            T_prev = T_target / cfg.T_scale
            P_prev = P_t

    arr_ar = np.array(pred_autoreg).squeeze()
    arr_os = np.array(pred_onestep).squeeze()
    arr_gt = np.array(ground_truth).squeeze()
    arr_dT_pred = np.array(delta_T_pred_list).squeeze()
    arr_dT_true = np.array(delta_T_true_list).squeeze()
    
    # Power profile
    P_profile = P_seq[0, :, 0].cpu().numpy()
    P_max_per_step = np.array([P_profile[t].max() for t in range(len(P_profile))])
    
    # Find hotspot (pixel with highest variance)
    var_map = np.var(arr_gt, axis=0)
    hy, hx = np.unravel_index(np.argmax(var_map), var_map.shape)
    
    # Find edge pixel
    ey, ex = 0, 0
    
    T_amb_val = data['T_amb'].item()
    h_val = data['h'].item()
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    steps = np.arange(len(arr_gt))
    
    # --- Panel 1: Hotspot Temperature ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, arr_gt[:, hy, hx], 'k-', linewidth=2, label='FDM (Truth)')
    ax1.plot(steps, arr_os[:, hy, hx], 'g--', linewidth=1.5, label='AI (One-Step)')
    ax1.plot(steps, arr_ar[:, hy, hx], 'r--', linewidth=1.5, label='AI (Autoreg)')
    ax1.axhline(y=T_amb_val, color='b', linestyle=':', alpha=0.5)
    ax1.set_title(f'Hotspot ({hx},{hy})', fontweight='bold')
    ax1.set_ylabel('Temperature [°C]')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Edge Temperature ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, arr_gt[:, ey, ex], 'k-', linewidth=2, label='FDM (Truth)')
    ax2.plot(steps, arr_os[:, ey, ex], 'g--', linewidth=1.5, label='AI (One-Step)')
    ax2.plot(steps, arr_ar[:, ey, ex], 'r--', linewidth=1.5, label='AI (Autoreg)')
    ax2.axhline(y=T_amb_val, color='b', linestyle=':', alpha=0.5)
    ax2.set_title(f'Edge ({ex},{ey})', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel 3: Power Profile ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, P_max_per_step, 'orange', linewidth=1.5)
    ax3.fill_between(steps, P_max_per_step, alpha=0.3, color='orange')
    ax3.set_title('Power Profile (max per step)', fontweight='bold')
    ax3.set_ylabel('Power Density [W/m³]')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # --- Panel 4: ΔT Comparison (One-Step) ---
    ax4 = fig.add_subplot(gs[1, 0])
    dT_pred_hot = arr_dT_pred[:, hy, hx]
    dT_true_hot = arr_dT_true[:, hy, hx]
    ax4.plot(steps, dT_true_hot, 'k-', linewidth=1.5, label='True ΔT')
    ax4.plot(steps, dT_pred_hot, 'g--', linewidth=1.5, label='Pred ΔT')
    ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_title('ΔT at Hotspot (One-Step)', fontweight='bold')
    ax4.set_ylabel('ΔT [°C]')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # --- Panel 5: Error Map (One-Step, last frame) ---
    ax5 = fig.add_subplot(gs[1, 1])
    error_map = arr_os[-1] - arr_gt[-1]
    im = ax5.imshow(error_map, cmap='RdBu_r', origin='lower',
                     vmin=-max(3, np.abs(error_map).max()),
                     vmax=max(3, np.abs(error_map).max()))
    ax5.set_title(f'Error Map (t={len(arr_gt)-1})', fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Error [°C]')
    
    # --- Panel 6: Temperature Map Comparison (last frame) ---
    ax6 = fig.add_subplot(gs[1, 2])
    vmin = min(arr_gt[-1].min(), arr_os[-1].min())
    vmax = max(arr_gt[-1].max(), arr_os[-1].max())
    
    ax6_left = ax6
    im_gt = ax6_left.imshow(arr_gt[-1], cmap='turbo', origin='lower', vmin=vmin, vmax=vmax)
    ax6_left.set_title(f'Ground Truth (t={len(arr_gt)-1})', fontweight='bold')
    plt.colorbar(im_gt, ax=ax6_left, label='T [°C]')
    
    # --- Panel 7: Mean Temperature Tracking ---
    ax7 = fig.add_subplot(gs[2, 0])
    gt_mean = arr_gt.mean(axis=(1,2))
    os_mean = arr_os.mean(axis=(1,2))
    ar_mean = arr_ar.mean(axis=(1,2))
    ax7.plot(steps, gt_mean, 'k-', linewidth=2, label='FDM Mean')
    ax7.plot(steps, os_mean, 'g--', linewidth=1.5, label='One-Step Mean')
    ax7.plot(steps, ar_mean, 'r--', linewidth=1.5, label='Autoreg Mean')
    ax7.set_title('Mean Temperature', fontweight='bold')
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('T [°C]')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)
    
    # --- Panel 8: Error Distribution ---
    ax8 = fig.add_subplot(gs[2, 1])
    os_errors = (arr_os - arr_gt).flatten()
    ax8.hist(os_errors, bins=50, color='green', alpha=0.7, label='One-Step Error', density=True)
    ax8.axvline(x=0, color='k', linestyle='--')
    ax8.set_title(f'Error Distribution (μ={os_errors.mean():.2f}, σ={os_errors.std():.2f}°C)', fontweight='bold')
    ax8.set_xlabel('Error [°C]')
    ax8.legend(fontsize=7)
    ax8.grid(True, alpha=0.3)
    
    # --- Panel 9: Metrics Summary ---
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    mae_os = np.abs(arr_os - arr_gt).mean()
    mae_ar = np.abs(arr_ar - arr_gt).mean()
    rmse_os = np.sqrt(((arr_os - arr_gt)**2).mean())
    rmse_ar = np.sqrt(((arr_ar - arr_gt)**2).mean())
    max_err_os = np.abs(arr_os - arr_gt).max()
    max_err_ar = np.abs(arr_ar - arr_gt).max()
    
    metrics_text = (
        f"Operating Conditions\n"
        f"{'─'*30}\n"
        f"h = {h_val:.0f} W/(m²K)\n"
        f"T_amb = {T_amb_val:.1f} °C\n"
        f"T_range = [{arr_gt.min():.1f}, {arr_gt.max():.1f}] °C\n\n"
        f"One-Step Accuracy\n"
        f"{'─'*30}\n"
        f"MAE  = {mae_os:.2f} °C\n"
        f"RMSE = {rmse_os:.2f} °C\n"
        f"Max  = {max_err_os:.2f} °C\n\n"
        f"Autoregressive Accuracy\n"
        f"{'─'*30}\n"
        f"MAE  = {mae_ar:.2f} °C\n"
        f"RMSE = {rmse_ar:.2f} °C\n"
        f"Max  = {max_err_ar:.2f} °C"
    )
    ax9.text(0.05, 0.95, metrics_text, transform=ax9.transAxes, fontsize=9,
             fontfamily='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle(f'Thermal Neural Surrogate — Diagnostic Report (Sequence #{idx})',
                 fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()
    
    return {'mae_os': mae_os, 'mae_ar': mae_ar, 'rmse_os': rmse_os, 'rmse_ar': rmse_ar}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--idx', type=int, default=10, help='Sequence index to visualize')
    parser.add_argument('--fresh', action='store_true', help='Use fresh untrained model')
    args = parser.parse_args()
    
    config = TrainingConfig()
    
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("Vis")
    
    print("Loading model...")
    model = create_model(verify_params=False).to(DEVICE)
    
    if not args.fresh:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        print(f"Model loaded (epoch {epoch})")
    else:
        print("Using fresh model (physics-only)")
    
    print("Loading data...")
    _, val_loader, _ = load_data(config, logger)
    
    metrics = visualize_diagnostic(model, val_loader.dataset, idx=args.idx)
    print(f"\nOne-Step MAE: {metrics['mae_os']:.2f}°C | Autoreg MAE: {metrics['mae_ar']:.2f}°C")
