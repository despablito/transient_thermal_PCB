"""
FDM vs Neural Surrogate — side-by-side animation
=================================================
Renders a hold-out drive-cycle sequence (generator seed outside the training
range) three ways: FDM ground truth, autoregressive surrogate prediction,
and their difference — plus a synchronized hotspot/power timeline.

Output: visualizations/fdm_vs_surrogate.gif (+ a static preview PNG).

Usage:
    python animate_comparison.py [--seed 70006] [--checkpoint weights/best_model.pth]
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from data_gen import ThermalProperties, TransientThermalSolver, PowerProfileGenerator
from model import create_model
from train import TrainingConfig

# Dark theme
BG = '#0e1420'
FG = '#e8edf4'
GRID = '#2a3550'
ACCENT_TRUTH = '#ff5c5c'
ACCENT_PRED = '#4dd8e6'
ACCENT_POWER = '#f2b134'


def simulate(seed: int, checkpoint: str):
    """Run FDM ground truth + autoregressive surrogate for one fresh sequence."""
    props = ThermalProperties()
    solver = TransientThermalSolver(props)
    cfg = TrainingConfig()

    # Same convention as benchmark_eval.generate_fresh_sequences: h/T_amb drawn
    # from a generator seeded with the sequence seed, profile from seed too.
    rng = np.random.default_rng(seed)
    h = rng.uniform(*props.h_range)
    T_amb = rng.uniform(*props.T_amb_range)

    pg = PowerProfileGenerator(props, seed=seed)
    P_seq, I_seq = pg.generate_power_sequence(props.n_frames)
    Q = pg.create_power_interpolator(P_seq, props.dt)
    _, T_hist = solver.solve(Q, h, T_amb, (0, props.dt * props.n_frames),
                             np.linspace(0, props.dt * props.n_frames, props.n_frames + 1))
    T_true = T_hist[1:]  # [100, H, W]

    # Surrogate, fully autoregressive from T_amb
    model = create_model(verify_params=False)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    H, W = props.ny, props.nx
    hm = torch.full((1, 1, H, W), h / cfg.h_scale, dtype=torch.float32)
    Ta = torch.full((1, 1, H, W), T_amb / cfg.T_scale, dtype=torch.float32)
    T_prev = torch.full((1, 1, H, W), T_amb / cfg.T_scale, dtype=torch.float32)
    P_prev = torch.zeros(1, 1, H, W)

    preds = []
    with torch.no_grad():
        for t in range(props.n_frames):
            P_t = torch.from_numpy(P_seq[t]).float().view(1, 1, H, W) / cfg.P_scale
            T_pred = T_prev + model(P_t, P_prev, T_prev, Ta, hm)
            preds.append(T_pred.numpy().squeeze() * cfg.T_scale)
            T_prev, P_prev = T_pred, P_t
    T_pred = np.array(preds)

    # Total nominal electrical power per frame [W] (for the timeline)
    sigma, th = pg.hotspot_sigma, props.thickness
    P_watts = P_seq.max(axis=(1, 2)) * (np.pi * sigma**2 * th)  # peak q_vol -> nominal W
    return T_true, T_pred, P_watts, h, T_amb, props.dt


def render(T_true, T_pred, P_watts, h, T_amb, dt, out_gif, out_png, fps=10):
    n = len(T_true)
    t_vec = np.arange(n) * dt
    err = T_pred - T_true
    mae_seq = np.abs(err).mean()

    vmin = min(T_true.min(), T_pred.min()) - 2
    vmax = max(T_true.max(), T_pred.max()) * 1.02
    emax = 25.0

    hy, hx = np.unravel_index(np.argmax(T_true.max(axis=0)), T_true.shape[1:])
    hot_true = T_true[:, hy, hx]
    hot_pred = T_pred[:, hy, hx]

    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
        'text.color': FG, 'axes.labelcolor': FG, 'axes.edgecolor': GRID,
        'xtick.color': FG, 'ytick.color': FG, 'font.size': 10,
    })

    fig = plt.figure(figsize=(12.8, 7.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.55, 1.0],
                  hspace=0.32, wspace=0.18, left=0.05, right=0.935, top=0.86, bottom=0.09)

    ax_t = fig.add_subplot(gs[0, 0]); ax_p = fig.add_subplot(gs[0, 1]); ax_e = fig.add_subplot(gs[0, 2])
    ax_ts = fig.add_subplot(gs[1, :])

    im_t = ax_t.imshow(T_true[0], cmap='turbo', origin='lower', vmin=vmin, vmax=vmax)
    im_p = ax_p.imshow(T_pred[0], cmap='turbo', origin='lower', vmin=vmin, vmax=vmax)
    im_e = ax_e.imshow(err[0], cmap='RdBu_r', origin='lower', vmin=-emax, vmax=emax)

    ax_t.set_title('FDM solver (ground truth)', color=FG, fontsize=11)
    ax_p.set_title('Neural surrogate — 48k params\n(autoregressive, own predictions only)', color=ACCENT_PRED, fontsize=11)
    err_title = ax_e.set_title('Error', color=FG, fontsize=11)
    for ax in (ax_t, ax_p, ax_e):
        ax.set_xticks([]); ax.set_yticks([])
    cb1 = fig.colorbar(im_p, ax=[ax_t, ax_p], fraction=0.046, pad=0.02); cb1.set_label('T [°C]', color=FG)
    cb2 = fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02); cb2.set_label('ΔT [°C]', color=FG)
    for cb in (cb1, cb2):
        cb.ax.yaxis.set_tick_params(color=FG)
        plt.setp(cb.ax.get_yticklabels(), color=FG)
        cb.outline.set_edgecolor(GRID)

    # Timeline
    ax_ts.plot(t_vec, hot_true, color=ACCENT_TRUTH, lw=2.2, label='Hotspot T — FDM')
    line_pred, = ax_ts.plot([], [], color=ACCENT_PRED, lw=2.0, ls='--', label='Hotspot T — surrogate')
    ax_pw = ax_ts.twinx()
    ax_pw.fill_between(t_vec, P_watts, color=ACCENT_POWER, alpha=0.18, step='post')
    ax_pw.plot(t_vec, P_watts, color=ACCENT_POWER, alpha=0.6, lw=1.0, drawstyle='steps-post')
    ax_pw.set_ylabel('Power (nominal) [W]', color=ACCENT_POWER)
    ax_pw.tick_params(axis='y', labelcolor=ACCENT_POWER, color=FG)
    ax_pw.set_ylim(0, max(P_watts.max() * 1.35, 10))
    for s in ax_pw.spines.values():
        s.set_color(GRID)
    cursor = ax_ts.axvline(0, color=FG, ls=':', alpha=0.6)
    ax_ts.set_xlim(0, t_vec[-1]); ax_ts.set_ylim(vmin, vmax * 1.02)
    ax_ts.set_xlabel('Time [s]'); ax_ts.set_ylabel('T [°C]')
    ax_ts.grid(True, color=GRID, alpha=0.5)
    ax_ts.legend(loc='upper right', fontsize=9, facecolor=BG, edgecolor=GRID, labelcolor=FG)

    fig.suptitle('SiC MOSFET junction temperature — physics solver vs 48k-parameter neural surrogate',
                 color=FG, fontsize=13.5, fontweight='bold', y=0.975)
    sub = (f'Hold-out drive cycle (never seen in training) | h = {h:.0f} W/(m²K) | '
           f'T_amb = {T_amb:.0f} °C | sequence MAE = {mae_seq:.1f} °C | real-time playback (dt = {dt:.1f} s)')
    fig.text(0.5, 0.925, sub, ha='center', color='#9fb0c8', fontsize=9.5)

    def update(f):
        im_t.set_array(T_true[f]); im_p.set_array(T_pred[f]); im_e.set_array(err[f])
        line_pred.set_data(t_vec[:f + 1], hot_pred[:f + 1])
        cursor.set_xdata([t_vec[f]])
        err_title.set_text(f'Error (frame MAE = {np.abs(err[f]).mean():.1f} °C)')
        return [im_t, im_p, im_e, line_pred, cursor, err_title]

    # Static preview (frame at peak temperature)
    update(int(np.argmax(hot_true)))
    fig.savefig(out_png, dpi=110)

    anim = FuncAnimation(fig, update, frames=n, interval=1000 // fps, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    print(f'Saved: {out_gif} + {out_png} | sequence MAE {mae_seq:.2f} °C')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=70006,
                    help='Fresh-sequence generator seed (outside training range 42-1041)')
    ap.add_argument('--checkpoint', default='weights/best_model.pth')
    ap.add_argument('--fps', type=int, default=10, help='10 fps = real-time (dt=0.1s)')
    args = ap.parse_args()
    assert args.seed > 1041, 'seed must be outside the training range'

    T_true, T_pred, P_watts, h, T_amb, dt = simulate(args.seed, args.checkpoint)
    render(T_true, T_pred, P_watts, h, T_amb, dt,
           'visualizations/fdm_vs_surrogate.gif',
           'visualizations/fdm_vs_surrogate_preview.png', fps=args.fps)
