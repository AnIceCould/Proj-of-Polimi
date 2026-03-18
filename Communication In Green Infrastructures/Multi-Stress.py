#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
水分胁迫前向验证工具 (最终严谨整合版)
1. 完全保留原始物理约束、数值稳定性处理与噪声模拟算法。
2. 嵌入 EC、pH 与 结构散射变量。
3. 拟合 Day 1 基准，并基于此生成假设性胁迫场景。
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import os

# ================= 物理常数 (保持原样) =================
c = 2.99792458e8
EPS_0 = 8.854187817e-12  # 真空介电常数
N_PTFE_REAL = 1.44
N_PTFE_LOSS = 0.0005


# ================= 1. 物理内核 (嵌入变量，保留逻辑) =================

def get_leaf_refractive_index(wc, f_hz, sigma=0.1, ph=7.0):
    """
    改进版: 正确处理复折射率，嵌入 EC(sigma) 和 pH 修正
    """
    # pH 对水分子弛豫特性的物理修正 (模拟氢键束缚)
    ph_dev = np.abs(ph - 7.0)
    eps_s_mod = 78.4 * (1 - 0.015 * ph_dev)
    tau_1_mod = 8.27e-12 * (1 + 0.01 * ph_dev)

    # 双Debye水模型
    eps_inf, eps_s, eps_1 = 4.9, eps_s_mod, 5.2
    tau_1, tau_2 = tau_1_mod, 0.18e-12
    omega = 2 * np.pi * f_hz

    eps_w = eps_inf + \
            (eps_s - eps_1) / (1 - 1j * omega * tau_1) + \
            (eps_1 - eps_inf) / (1 - 1j * omega * tau_2)

    # 嵌入电导率项 (EC)
    eps_sigma = 1j * sigma / (omega * EPS_0)
    eps_w_total = eps_w + eps_sigma

    # LLL混合公式
    eps_dry = complex(2.5, 0.05)
    term_w = wc * (eps_w_total ** (1 / 3))
    term_d = (1 - wc) * (eps_dry ** (1 / 3))
    eps_eff = (term_w + term_d) ** 3

    # 正确的复折射率计算
    n_complex = np.sqrt(eps_eff)

    # 确保物理意义: n = n' - i*n'' (n'' > 0 表示吸收)
    n_real = np.real(n_complex)
    n_imag = np.abs(np.imag(n_complex))

    return complex(n_real, -n_imag)


def transfer_matrix_solver(f_hz, d_leaf_mm, wc, d_ptfe_mm, sigma, ph, phase_noise=0.0):
    """
    原始版本逻辑: 增强数值稳定性 + 支持相位噪声
    """
    k0 = 2 * np.pi * f_hz / c
    n_air = 1.0 + 0j
    n_ptfe = complex(N_PTFE_REAL, -N_PTFE_LOSS)
    n_leaf = get_leaf_refractive_index(wc, f_hz, sigma, ph)

    d_ptfe = d_ptfe_mm * 1e-3
    d_leaf = d_leaf_mm * 1e-3

    def interface(n1, n2):
        r = (n1 - n2) / (n1 + n2)
        t = 2 * n1 / (n1 + n2)
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def prop(n, d, add_phase_noise=False):
        phase = -1j * n * k0 * d
        # 防止指数溢出 (保留原始阈值)
        phase = np.clip(phase.real, -50, 50) + 1j * phase.imag

        # 添加相位噪声
        if add_phase_noise and phase_noise != 0.0:
            phase += 1j * phase_noise

        p = np.exp(phase)
        return np.array([[1 / p, 0], [0, p]], dtype=complex)

    # 构建完整转移矩阵 (保留原始结构)
    M = interface(n_air, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_leaf) @ prop(n_leaf, d_leaf, add_phase_noise=True) @ \
        interface(n_leaf, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_air)

    # 透射系数
    T = np.abs(1.0 / M[0, 0]) ** 2
    return np.clip(T, 1e-15, 1.0)


def forward_solver(params, f_hz, phase_noise_array=None):
    """
    前向求解器: 嵌入散射项，保持原始物理流程
    params: [wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff]
    """
    p = list(params)
    while len(p) < 7:
        if len(p) == 4: p.append(0.1)  # sigma
        if len(p) == 5: p.append(7.0)  # ph
        if len(p) == 6: p.append(0.0)  # scat_coeff

    wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff = p

    if phase_noise_array is None:
        phase_noise_array = np.zeros(len(f_hz))

    T = [transfer_matrix_solver(f, d_leaf, wc, d_ptfe, sigma, ph, pn)
         for f, pn in zip(f_hz, phase_noise_array)]

    y_db = 10 * np.log10(np.maximum(np.array(T), 1e-15)) + offset

    # 嵌入结构散射损耗 S * f^2 (导致斜率变陡)
    f_thz = f_hz / 1e12
    return y_db - (scat_coeff * (f_thz ** 2))


# ================= 2. 噪声模拟模块 (原封不动还原) =================

class NoiseSimulator:
    """
    还原原始所有噪声细节
    """

    @staticmethod
    def detector_noise(signal, snr_db=30):
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return noise

    @staticmethod
    def frequency_dependent_noise(f_hz, amplitude=0.3):
        f_thz = f_hz / 1e12
        noise_profile = amplitude * (1 + 0.5 * (f_thz - 0.75))
        return np.random.normal(0, noise_profile)

    @staticmethod
    def systematic_drift(f_hz, drift_amplitude=0.2):
        f_thz = f_hz / 1e12
        drift = drift_amplitude * np.sin(2 * np.pi * (f_thz - 0.75) / 0.35)
        return drift

    @staticmethod
    def etalon_effect(f_hz, period_thz=0.05, amplitude=0.15):
        f_thz = f_hz / 1e12
        ripple = amplitude * np.sin(2 * np.pi * f_thz / period_thz)
        return ripple

    @staticmethod
    def sample_positioning_error(signal, position_std=0.05):
        phase_noise = np.random.normal(0, position_std, len(signal))
        return phase_noise

    @staticmethod
    def phase_noise_turbulence(f_hz, std_radians=0.05):
        return np.random.normal(0, std_radians, len(f_hz))

    @staticmethod
    def phase_noise_dispersion(f_hz, dispersion_factor=0.02):
        return dispersion_factor * (f_hz / 1e12 - 0.75) * np.random.randn(len(f_hz))

    @staticmethod
    def phase_noise_correlated(f_hz, correlation_length=0.1, amplitude=0.03):
        n_points = len(f_hz)
        base = np.random.randn(n_points)
        smooth_noise = gaussian_filter1d(base, sigma=n_points * correlation_length)
        return amplitude * smooth_noise

    @staticmethod
    def generate_phase_noise(f_hz, noise_level='medium'):
        phase_params = {
            'low': {'turb': 0.02, 'disp': 0.01, 'corr': 0.015},
            'medium': {'turb': 0.05, 'disp': 0.02, 'corr': 0.03},
            'high': {'turb': 0.10, 'disp': 0.04, 'corr': 0.05}
        }
        p = phase_params[noise_level]
        phase_noise = NoiseSimulator.phase_noise_turbulence(f_hz, p['turb'])
        phase_noise += NoiseSimulator.phase_noise_dispersion(f_hz, p['disp'])
        phase_noise += NoiseSimulator.phase_noise_correlated(f_hz, amplitude=p['corr'])
        return phase_noise

    @staticmethod
    def add_realistic_noise(signal, f_hz, params, noise_level='medium', include_phase=True):
        amp_noise_params = {
            'low': {'snr': 40, 'freq': 0.15, 'drift': 0.1, 'etalon': 0.1, 'pos': 0.03},
            'medium': {'snr': 30, 'freq': 0.30, 'drift': 0.2, 'etalon': 0.15, 'pos': 0.05},
            'high': {'snr': 25, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08}
        }
        p = amp_noise_params[noise_level]

        if include_phase:
            phase_noise = NoiseSimulator.generate_phase_noise(f_hz, noise_level)
            noisy_signal = forward_solver(params, f_hz, phase_noise)
        else:
            noisy_signal = signal.copy()

        noisy_signal += NoiseSimulator.detector_noise(signal, p['snr'])
        noisy_signal += NoiseSimulator.frequency_dependent_noise(f_hz, p['freq'])
        noisy_signal += NoiseSimulator.systematic_drift(f_hz, p['drift'])
        noisy_signal += NoiseSimulator.etalon_effect(f_hz, amplitude=p['etalon'])
        noisy_signal += NoiseSimulator.sample_positioning_error(signal, p['pos'])
        return noisy_signal


# ================= 3. 复杂胁迫模型 =================

def apply_comprehensive_stress(p_d1, wc_s, d_s, sigma_s, ph_v, scat_v, off_d):
    """
    p_d1: [WC, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff]
    """
    return [
        p_d1[0] * wc_s,  # 水分
        p_d1[1] * d_s,  # 厚度
        p_d1[2],  # PTFE
        p_d1[3] + off_d,  # Offset漂移
        p_d1[4] * sigma_s,  # EC缩放
        ph_v,  # pH设置
        scat_v  # 散射设置
    ]


# ================= 4. 数据处理 =================

def load_data(path):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path, header=None, names=['Freq', 'Db']).groupby('Freq', as_index=False).mean()
    f = np.linspace(0.78, 0.97, 100) * 1e12
    interp = interp1d(df['Freq'] * 1e12, df['Db'], kind='cubic', fill_value="extrapolate")
    return f, interp(f)


# ================= 5. 主程序 =================

def main():
    print("=" * 70)
    print("THz叶片多维胁迫仿真器 (严格还原物理逻辑版)")
    print("=" * 70)

    # 1. 加载 Day 1 实验数据
    f_hz, y_d1 = load_data("targets/Spinach_Day1.csv")
    if y_d1 is None: return
    f_thz = f_hz / 1e12

    # 2. 拟合 Day 1 基准参数 (带原始物理约束)
    print("\n[步骤 1] 正在拟合 Day 1 物理基准状态...")

    def objective(p):
        # 还原物理约束
        if not (0.6 < p[0] < 0.97 and 0.1 < p[1] < 0.8 and 1.0 < p[2] < 3.0):
            return 1e6
        return np.mean((y_d1 - forward_solver(list(p) + [0.1, 7.0, 0.0], f_hz)) ** 2)

    result = minimize(objective, x0=[0.85, 0.4, 2.0, -2.0], method='Nelder-Mead', options={'maxiter': 500})
    p_day1 = list(result.x) + [0.1, 7.0, 0.0]

    print(f"\nDay 1 拟合结论:")
    print(f"  含水量 WC  = {p_day1[0] * 100:.2f} %")
    print(f"  厚度 d     = {p_day1[1] * 1000:.2f} μm")
    print(f"  PTFE厚度   = {p_day1[2]:.3f} mm")
    print(f"  系统偏移   = {p_day1[3]:.2f} dB")
    print(f"  拟合误差   = {result.fun:.4f}")

    # 3. 定义假设性场景 (WC缩放, d缩放, EC缩放, pH, 散射S, Offset漂移, 标签, 颜色)
    scenarios = [
        (0.75, 0.65, 1.0, 7.0, 0.0, 0.0, 'Hypo A: Drought Stress', 'orange'),
        (0.95, 0.90, 30.0, 7.0, 0.0, 0.0, 'Hypo B: Salt Stress (EC)', 'blue'),
        (0.70, 0.60, 1.0, 7.0, 15.0, 0.0, 'Hypo C: Structural Stress', 'red'),
        (0.95, 0.90, 1.0, 3.0, 0.0, 0.0, 'Hypo D: Metabolic Stress (pH)', 'purple'),
    ]

    # 4. 可视化
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Simulated Stress Scenarios (Based on Day 1 Fit & Realistic Noise)', fontsize=15, fontweight='bold')

    # 子图1: Day 1 展示 (实验 vs 理想拟合 vs 带噪模拟)
    ax1 = axes[0, 0]
    y_d1_ideal = forward_solver(p_day1, f_hz)
    y_d1_noisy = NoiseSimulator.add_realistic_noise(y_d1_ideal, f_hz, p_day1, 'medium')

    ax1.scatter(f_thz, y_d1, c='gray', alpha=0.4, label='Exp. Day 1')
    # ax1.plot(f_thz, y_d1_ideal, 'k-', lw=1.5, label='Ideal Fit')
    ax1.plot(f_thz, y_d1_noisy, 'g--', lw=1.5, label='Noisy Sim Day 1')
    ax1.set_title("Day 1 Baseline (With Noise)")
    ax1.set_ylim([np.min(y_d1) - 5, np.max(y_d1) + 5])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 子图2-4: 假设胁迫场景 (遗忘 Day 4 数据)
    for idx, (wc_s, d_s, sig_s, ph_v, scat_v, off_d, label, col) in enumerate(scenarios):
        ax = axes.flat[idx + 1]
        p_s = apply_comprehensive_stress(p_day1, wc_s, d_s, sig_s, ph_v, scat_v, off_d)

        y_ideal = forward_solver(p_s, f_hz)
        y_noisy = NoiseSimulator.add_realistic_noise(y_ideal, f_hz, p_s, 'medium')

        # ax.plot(f_thz, y_ideal, color=col, alpha=0.3, label='Ideal Scenario')
        ax.plot(f_thz, y_noisy, color=col, ls='--', lw=2, label='Noisy Scenario')

        slope = (y_ideal[-1] - y_ideal[0]) / (f_thz[-1] - f_thz[0])
        ax.set_title(f"{label}\nSlope: {slope:.2f} dB/THz")
        ax.set_ylim([np.min(y_ideal) - 8, np.max(y_ideal) + 8])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()