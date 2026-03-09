"""
Benchmark: 50 losowych sekwencji testowych
==========================================
Porównanie wyjścia FDM (benchmark) z predykcjami wytrenowanej sieci.
Raport: MAE, RMSE, Max Error + przedziały ufności 95%.
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from scipy import stats

from model import create_model
from train import TrainingConfig, load_data

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEED = 42
N_SEQUENCES = 50


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_sequence(model, data, cfg, device):
    """Dla jednej sekwencji zwraca MAE, RMSE, Max (one-step i autoregresyjnie)."""
    model.eval()
    T_seq = data['T_seq'].unsqueeze(0).to(device)
    P_seq = data['P_seq'].unsqueeze(0).to(device)
    T_initial = data['T_initial'].unsqueeze(0).to(device)
    P_initial = data['P_initial'].unsqueeze(0).to(device)
    h = data['h'].unsqueeze(0).to(device)
    T_amb = data['T_amb'].unsqueeze(0).to(device)
    H, W = T_seq.shape[-2], T_seq.shape[-1]
    h_map = h.view(1, 1, 1, 1).expand(1, 1, H, W) / cfg.h_scale
    Tamb_map = T_amb.view(1, 1, 1, 1).expand(1, 1, H, W) / cfg.T_scale
    arr_gt = T_seq.cpu().numpy().squeeze()

    # Autoregresja: T_prev = własna predykcja
    pred_autoreg = []
    T_prev = T_initial / cfg.T_scale
    P_prev = P_initial / cfg.P_scale
    with torch.no_grad():
        for t in range(T_seq.shape[1]):
            P_t = P_seq[:, t] / cfg.P_scale
            delta_T = model(P_t, P_prev, T_prev, Tamb_map, h_map)
            T_pred = T_prev + delta_T
            pred_autoreg.append((T_pred.cpu().numpy() * cfg.T_scale).squeeze())
            T_prev = T_pred
            P_prev = P_t
    arr_ar = np.array(pred_autoreg)

    # One-step: T_prev = ground truth (teacher forcing)
    pred_onestep = []
    T_prev = T_initial / cfg.T_scale
    P_prev = P_initial / cfg.P_scale
    with torch.no_grad():
        for t in range(T_seq.shape[1]):
            P_t = P_seq[:, t] / cfg.P_scale
            T_target = T_seq[:, t]
            delta_T = model(P_t, P_prev, T_prev, Tamb_map, h_map)
            T_pred = T_prev + delta_T
            pred_onestep.append((T_pred.cpu().numpy() * cfg.T_scale).squeeze())
            T_prev = T_target / cfg.T_scale
            P_prev = P_t
    arr_os = np.array(pred_onestep)

    err_os = arr_os - arr_gt
    err_ar = arr_ar - arr_gt
    mae_os = np.abs(err_os).mean()
    mae_ar = np.abs(err_ar).mean()
    rmse_os = np.sqrt((err_os ** 2).mean())
    rmse_ar = np.sqrt((err_ar ** 2).mean())
    max_os = np.abs(err_os).max()
    max_ar = np.abs(err_ar).max()
    return {
        'mae_one_step': float(mae_os),
        'mae_autoreg': float(mae_ar),
        'rmse_one_step': float(rmse_os),
        'rmse_autoreg': float(rmse_ar),
        'max_one_step': float(max_os),
        'max_autoreg': float(max_ar),
    }


def confidence_interval(values, confidence=0.95):
    """Przedział ufności średniej (t-Student)."""
    n = len(values)
    if n < 2:
        return np.mean(values), 0.0, np.mean(values), np.mean(values)
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sem
    return mean, margin, mean - margin, mean + margin


def main():
    parser = argparse.ArgumentParser(description='Benchmark na 50 losowych sekwencjach testowych')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', help='Ścieżka do modelu')
    parser.add_argument('--n', type=int, default=N_SEQUENCES, help='Liczba sekwencji (domyślnie 50)')
    parser.add_argument('--seed', type=int, default=SEED, help='Ziarno losowości')
    parser.add_argument('--out', default='benchmark_results.json', help='Plik JSON z wynikami')
    parser.add_argument('--no-plot', action='store_true', help='Bez generowania wykresu')
    args = parser.parse_args()

    set_seed(args.seed)
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("Benchmark")

    config = TrainingConfig()
    print("Ładowanie modelu...")
    model = create_model(verify_params=False).to(DEVICE)
    cp = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    epoch = cp.get('epoch', '?')
    print(f"  Model załadowany (epoch {epoch})")

    print("Ładowanie danych (podział train/val/test)...")
    _, _, test_loader = load_data(config, logger)
    dataset = test_loader.dataset
    n_seq = dataset.n_sequences
    n_eval = min(args.n, n_seq)
    rng = np.random.default_rng(args.seed)
    seq_indices = rng.choice(n_seq, size=n_eval, replace=False)
    starts = dataset.starts_per_seq

    print(f"  Zbiór testowy: {n_seq} sekwencji. Ewaluacja: {n_eval} losowych sekwencji.\n")
    results_per_seq = []

    for i, seq_i in enumerate(seq_indices):
        idx = int(seq_i * starts)
        data = dataset[idx]
        res = evaluate_sequence(model, data, config, DEVICE)
        res['sequence_id'] = int(seq_i)
        results_per_seq.append(res)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Sekwencja {i+1}/{n_eval} (id={seq_i}): MAE_os={res['mae_one_step']:.2f}°C, MAE_ar={res['mae_autoreg']:.2f}°C")

    # Agregacja
    mae_os = np.array([r['mae_one_step'] for r in results_per_seq])
    mae_ar = np.array([r['mae_autoreg'] for r in results_per_seq])
    rmse_os = np.array([r['rmse_one_step'] for r in results_per_seq])
    rmse_ar = np.array([r['rmse_autoreg'] for r in results_per_seq])
    max_os = np.array([r['max_one_step'] for r in results_per_seq])
    max_ar = np.array([r['max_autoreg'] for r in results_per_seq])

    ci = 0.95
    mean_mae_os, margin_mae_os, lo_mae_os, hi_mae_os = confidence_interval(mae_os, ci)
    mean_mae_ar, margin_mae_ar, lo_mae_ar, hi_mae_ar = confidence_interval(mae_ar, ci)
    mean_rmse_os, margin_rmse_os, lo_rmse_os, hi_rmse_os = confidence_interval(rmse_os, ci)
    mean_rmse_ar, margin_rmse_ar, lo_rmse_ar, hi_rmse_ar = confidence_interval(rmse_ar, ci)
    mean_max_os, margin_max_os, lo_max_os, hi_max_os = confidence_interval(max_os, ci)
    mean_max_ar, margin_max_ar, lo_max_ar, hi_max_ar = confidence_interval(max_ar, ci)

    summary = {
        'n_sequences': n_eval,
        'seed': args.seed,
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'confidence': ci,
        'one_step': {
            'MAE': {'mean': float(mean_mae_os), 'std': float(np.std(mae_os)), 'ci95_low': float(lo_mae_os), 'ci95_high': float(hi_mae_os)},
            'RMSE': {'mean': float(mean_rmse_os), 'std': float(np.std(rmse_os)), 'ci95_low': float(lo_rmse_os), 'ci95_high': float(hi_rmse_os)},
            'Max': {'mean': float(mean_max_os), 'std': float(np.std(max_os)), 'ci95_low': float(lo_max_os), 'ci95_high': float(hi_max_os)},
        },
        'autoregressive': {
            'MAE': {'mean': float(mean_mae_ar), 'std': float(np.std(mae_ar)), 'ci95_low': float(lo_mae_ar), 'ci95_high': float(hi_mae_ar)},
            'RMSE': {'mean': float(mean_rmse_ar), 'std': float(np.std(rmse_ar)), 'ci95_low': float(lo_rmse_ar), 'ci95_high': float(hi_rmse_ar)},
            'Max': {'mean': float(mean_max_ar), 'std': float(np.std(max_ar)), 'ci95_low': float(lo_max_ar), 'ci95_high': float(hi_max_ar)},
        },
        'per_sequence': results_per_seq,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nZapisano: {out_path}")

    # Raport tekstowy
    print("\n" + "=" * 70)
    print("BENCHMARK: FDM (benchmark) vs sieć neuronowa — podsumowanie")
    print("=" * 70)
    print(f"Liczba sekwencji: {n_eval}  |  Przedział ufności: {ci*100:.0f}%  |  Ziarno: {args.seed}")
    print("-" * 70)
    print("One-Step (predykcja z poprawnym T_{t-1}):")
    print(f"  MAE:  {mean_mae_os:.2f} °C  (95% CI: [{lo_mae_os:.2f}, {hi_mae_os:.2f}])  σ = {np.std(mae_os):.2f}")
    print(f"  RMSE: {mean_rmse_os:.2f} °C  (95% CI: [{lo_rmse_os:.2f}, {hi_rmse_os:.2f}])")
    print(f"  Max:  {mean_max_os:.2f} °C  (95% CI: [{lo_max_os:.2f}, {hi_max_os:.2f}])")
    print("-" * 70)
    print("Autoregresja (predykcja z własnym T_{t-1}):")
    print(f"  MAE:  {mean_mae_ar:.2f} °C  (95% CI: [{lo_mae_ar:.2f}, {hi_mae_ar:.2f}])  σ = {np.std(mae_ar):.2f}")
    print(f"  RMSE: {mean_rmse_ar:.2f} °C  (95% CI: [{lo_rmse_ar:.2f}, {hi_rmse_ar:.2f}])")
    print(f"  Max:  {mean_max_ar:.2f} °C  (95% CI: [{lo_max_ar:.2f}, {hi_max_ar:.2f}])")
    print("=" * 70)

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # Boxplot MAE
            ax = axes[0]
            bp = ax.boxplot([mae_os, mae_ar], tick_labels=['One-Step', 'Autoregresja'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax.set_ylabel('MAE [°C]')
            ax.set_title('Rozkład MAE (50 sekwencji)')
            ax.grid(True, alpha=0.3)
            # Słupek średniej z 95% CI
            ax = axes[1]
            x = [0, 1]
            means = [mean_mae_os, mean_mae_ar]
            errs = [margin_mae_os, margin_mae_ar]
            colors = ['green', 'coral']
            bars = ax.bar(x, means, yerr=errs, capsize=8, color=colors, alpha=0.8, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(['One-Step', 'Autoregresja'])
            ax.set_ylabel('MAE [°C]')
            ax.set_title(f'Średni MAE z 95% przedziałem ufności (n={n_eval})')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plot_path = Path('visualizations/benchmark_50seq.png')
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Wykres zapisany: {plot_path}")
        except Exception as e:
            print(f"Wykres nie został wygenerowany: {e}")

    return summary


if __name__ == '__main__':
    main()
