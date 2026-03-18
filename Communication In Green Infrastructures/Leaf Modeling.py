#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
水分胁迫前向验证工具 (带真实噪声)

改进点:
1. 修复复折射率处理
2. 添加多层次真实噪声模拟 (振幅 + 相位)
3. 提高数值稳定性
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os

# ================= 物理常数 =================
c = 2.99792458e8
N_PTFE_REAL = 1.44
N_PTFE_LOSS = 0.0005


# ================= 1. 改进的物理内核 =================

def get_leaf_refractive_index(wc, f_hz):
    """
    改进版: 正确处理复折射率
    """
    # 双Debye水模型
    eps_inf, eps_s, eps_1 = 4.9, 78.4, 5.2
    tau_1, tau_2 = 8.27e-12, 0.18e-12
    omega = 2 * np.pi * f_hz

    eps_w = eps_inf + \
            (eps_s - eps_1) / (1 - 1j * omega * tau_1) + \
            (eps_1 - eps_inf) / (1 - 1j * omega * tau_2)

    # LLL混合公式
    eps_dry = complex(2.5, 0.05)
    term_w = wc * (eps_w ** (1 / 3))
    term_d = (1 - wc) * (eps_dry ** (1 / 3))
    eps_eff = (term_w + term_d) ** 3

    # 正确的复折射率计算
    n_complex = np.sqrt(eps_eff)

    # 确保物理意义: n = n' - i*n'' (n'' > 0 表示吸收)
    n_real = np.real(n_complex)
    n_imag = np.abs(np.imag(n_complex))

    return complex(n_real, -n_imag)


def transfer_matrix_solver(f_hz, d_leaf_mm, wc, d_ptfe_mm, phase_noise=0.0):
    """
    改进版: 增强数值稳定性 + 支持相位噪声

    phase_noise: 额外的相位误差 (弧度)
    """
    k0 = 2 * np.pi * f_hz / c
    n_air = 1.0 + 0j
    n_ptfe = complex(N_PTFE_REAL, -N_PTFE_LOSS)
    n_leaf = get_leaf_refractive_index(wc, f_hz)

    d_ptfe = d_ptfe_mm * 1e-3
    d_leaf = d_leaf_mm * 1e-3

    def interface(n1, n2):
        r = (n1 - n2) / (n1 + n2)
        t = 2 * n1 / (n1 + n2)
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def prop(n, d, add_phase_noise=False):
        phase = -1j * n * k0 * d
        # 防止指数溢出
        phase = np.clip(phase.real, -50, 50) + 1j * phase.imag

        # 如果需要,添加相位噪声
        if add_phase_noise and phase_noise != 0.0:
            phase += 1j * phase_noise

        p = np.exp(phase)
        return np.array([[1 / p, 0], [0, p]], dtype=complex)

    # 构建完整转移矩阵,在叶片层添加相位噪声
    M = interface(n_air, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_leaf) @ prop(n_leaf, d_leaf, add_phase_noise=True) @ \
        interface(n_leaf, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_air)

    # 透射系数
    T = np.abs(1.0 / M[0, 0]) ** 2
    return np.clip(T, 1e-15, 1.0)


def forward_solver(params, f_hz, phase_noise_array=None):
    """
    前向求解器 (支持相位噪声)

    phase_noise_array: 每个频率点的相位噪声数组 (弧度)
    """
    wc, d_leaf, d_ptfe, offset = params

    if phase_noise_array is None:
        phase_noise_array = np.zeros(len(f_hz))

    T = [transfer_matrix_solver(f, d_leaf, wc, d_ptfe, pn)
         for f, pn in zip(f_hz, phase_noise_array)]

    return 10 * np.log10(np.maximum(np.array(T), 1e-15)) + offset


# ================= 2. 噪声模拟模块 =================

class NoiseSimulator:
    """
    模拟THz-TDS测量中的各类噪声
    包括振幅噪声和相位噪声
    """

    @staticmethod
    def detector_noise(signal, snr_db=30):
        """
        探测器噪声 (高斯白噪声)
        典型信噪比: 25-40 dB
        """
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return noise

    @staticmethod
    def frequency_dependent_noise(f_hz, amplitude=0.3):
        """
        频率相关噪声 (高频衰减)
        物理来源: 水汽吸收、高频信噪比下降
        """
        f_thz = f_hz / 1e12
        # 高频端噪声增强
        noise_profile = amplitude * (1 + 0.5 * (f_thz - 0.75))
        return np.random.normal(0, noise_profile)

    @staticmethod
    def systematic_drift(f_hz, drift_amplitude=0.2):
        """
        系统漂移 (低频基线漂移)
        物理来源: 温度漂移、对准误差
        """
        f_thz = f_hz / 1e12
        # 低频正弦漂移
        drift = drift_amplitude * np.sin(2 * np.pi * (f_thz - 0.75) / 0.35)
        return drift

    @staticmethod
    def etalon_effect(f_hz, period_thz=0.05, amplitude=0.15):
        """
        法布里-珀罗干涉效应 (Etalon)
        物理来源: 多次反射产生的周期性波纹
        """
        f_thz = f_hz / 1e12
        ripple = amplitude * np.sin(2 * np.pi * f_thz / period_thz)
        return ripple

    @staticmethod
    def sample_positioning_error(signal, position_std=0.05):
        """
        样品定位误差 (振幅噪声)
        物理来源: 光程差变化
        """
        phase_noise = np.random.normal(0, position_std, len(signal))
        return phase_noise

    @staticmethod
    def phase_noise_turbulence(f_hz, std_radians=0.05):
        """
        湍流相位噪声

        物理来源:
        - 空气折射率波动
        - 机械振动
        - 光束指向不稳定性

        返回: 相位噪声 (弧度)
        """
        # 白相位噪声
        base_noise = np.random.normal(0, std_radians, len(f_hz))
        return base_noise

    @staticmethod
    def phase_noise_dispersion(f_hz, dispersion_factor=0.02):
        """
        频率相关的相位噪声 (色散)

        物理来源:
        - 材料色散变化
        - 厚度不均匀性
        - 温度梯度

        返回: 相位噪声 (弧度)
        """
        f_thz = f_hz / 1e12
        # 高频累积更多相位误差
        dispersion = dispersion_factor * (f_thz - 0.75) * np.random.randn(len(f_hz))
        return dispersion

    @staticmethod
    def phase_noise_correlated(f_hz, correlation_length=0.1, amplitude=0.03):
        """
        相关相位噪声 (低频相位漂移)

        物理来源:
        - 缓慢热膨胀
        - 机械蠕变
        - 湿度变化

        返回: 相位噪声 (弧度)
        """
        f_thz = f_hz / 1e12
        # 使用低频成分生成平滑的相关噪声
        n_points = len(f_hz)
        # 创建基础噪声
        base = np.random.randn(n_points)
        # 应用平滑
        from scipy.ndimage import gaussian_filter1d
        smooth_noise = gaussian_filter1d(base, sigma=n_points * correlation_length)
        return amplitude * smooth_noise

    @staticmethod
    def generate_phase_noise(f_hz, noise_level='medium'):
        """
        生成综合相位噪声

        返回: 总相位噪声数组 (弧度)
        """
        phase_params = {
            'low': {'turb': 0.02, 'disp': 0.01, 'corr': 0.015},
            'medium': {'turb': 0.05, 'disp': 0.02, 'corr': 0.03},
            'high': {'turb': 0.10, 'disp': 0.04, 'corr': 0.05}
        }

        p = phase_params[noise_level]

        phase_noise = np.zeros(len(f_hz))
        phase_noise += NoiseSimulator.phase_noise_turbulence(f_hz, p['turb'])
        phase_noise += NoiseSimulator.phase_noise_dispersion(f_hz, p['disp'])
        phase_noise += NoiseSimulator.phase_noise_correlated(f_hz, amplitude=p['corr'])

        return phase_noise

    @staticmethod
    def add_realistic_noise(signal, f_hz, params, noise_level='medium', include_phase=True):
        """
        组合噪声模型 (振幅 + 相位)

        noise_level: 'low', 'medium', 'high'
        include_phase: 如果为True,则用相位噪声重新生成信号
        """
        amp_noise_params = {
            'low': {'snr': 40, 'freq': 0.15, 'drift': 0.1, 'etalon': 0.1, 'pos': 0.03},
            'medium': {'snr': 30, 'freq': 0.30, 'drift': 0.2, 'etalon': 0.15, 'pos': 0.05},
            'high': {'snr': 25, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08}
        }

        p = amp_noise_params[noise_level]

        # 如果包含相位噪声,从头重新生成信号
        if include_phase:
            phase_noise = NoiseSimulator.generate_phase_noise(f_hz, noise_level)
            noisy_signal = forward_solver(params, f_hz, phase_noise)
        else:
            noisy_signal = signal.copy()

        # 添加振幅噪声
        noisy_signal += NoiseSimulator.detector_noise(signal, p['snr'])
        noisy_signal += NoiseSimulator.frequency_dependent_noise(f_hz, p['freq'])
        noisy_signal += NoiseSimulator.systematic_drift(f_hz, p['drift'])
        noisy_signal += NoiseSimulator.etalon_effect(f_hz, amplitude=p['etalon'])
        noisy_signal += NoiseSimulator.sample_positioning_error(signal, p['pos'])

        return noisy_signal


# ================= 3. 水分胁迫模型 =================

def apply_water_stress(params_day1, wc_scale, d_scale, offset_drift):
    """
    施加水分胁迫变化

    params_day1: [WC, d_leaf, d_ptfe, offset]
    wc_scale:    水分保持率 (0-1)
    d_scale:     厚度保持率 (0-1)
    offset_drift: 系统偏移变化 (dB)
    """
    wc_0, d_0, d_ptfe, off_0 = params_day1

    return [
        wc_0 * wc_scale,  # 水分变化
        d_0 * d_scale,  # 厚度变化
        d_ptfe,  # PTFE不变
        off_0 + offset_drift  # Offset微调
    ]


# ================= 4. 数据处理 =================

def load_data(path):
    """加载实验数据"""
    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path, header=None, names=['Freq', 'Db'])
    df = df.groupby('Freq', as_index=False).mean().sort_values('Freq')

    # 插值到统一频率网格
    f = np.linspace(0.78, 0.97, 100) * 1e12
    interp = interp1d(df['Freq'] * 1e12, df['Db'],
                      kind='cubic', fill_value="extrapolate")

    return f, interp(f)


# ================= 5. 主程序 =================

def main():
    print("=" * 70)
    print("THz叶片水分胁迫模拟器 - 带真实噪声 (振幅 + 相位)")
    print("=" * 70)

    # 1. 加载数据
    f_hz, y_d1 = load_data("targets/Spinach_Day1.csv")
    _, y_d4 = load_data("targets/Spinach_Day4.csv")

    if y_d1 is None:
        print("错误: 未找到数据文件!")
        return

    f_thz = f_hz / 1e12

    # 2. 拟合 Day 1 基准参数
    print("\n[步骤 1] 拟合 Day 1 基准状态...")

    def objective(p):
        # 物理约束
        if not (0.6 < p[0] < 0.97 and
                0.1 < p[1] < 0.8 and
                1.0 < p[2] < 3.0):
            return 1e6
        return np.mean((y_d1 - forward_solver(p, f_hz)) ** 2)

    # 优化
    result = minimize(objective,
                      x0=[0.85, 0.4, 2.0, -2.0],
                      method='Nelder-Mead',
                      options={'maxiter': 500})

    p_day1 = result.x

    print(f"\nDay 1 拟合参数:")
    print(f"  含水量 WC  = {p_day1[0] * 100:.2f} %")
    print(f"  厚度 d     = {p_day1[1] * 1000:.2f} μm")
    print(f"  PTFE厚度   = {p_day1[2]:.3f} mm")
    print(f"  系统偏移   = {p_day1[3]:.2f} dB")
    print(f"  拟合误差   = {result.fun:.4f}")

    # 3. 生成带噪声的模拟数据
    print("\n[步骤 2] 生成带噪声的胁迫场景...")

    # 定义测试场景
    scenarios = [
        # (WC保持率, 厚度保持率, Offset漂移, 噪声等级, 标签, 颜色)
        (0.75, 0.65, -0.00, 'medium', 'Scenario 2: Moderate Stress + Medium Noise', 'orange'),
    ]

    # 4. 可视化
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('THz Leaf Stress Simulation',
                 fontsize=14, fontweight='bold')

    # 子图1: Day 1 理想 vs 带噪声
    ax1 = axes[0]
    y_d1_clean = forward_solver(p_day1, f_hz)
    y_d1_noisy = NoiseSimulator.add_realistic_noise(y_d1_clean, f_hz, p_day1, 'medium')

    ax1.scatter(f_thz, y_d1, c='gray', alpha=0.3, s=20, label='Exp. Data')
    # ax1.plot(f_thz, y_d1_clean, 'g-', lw=2, label='Ideal Fit')
    ax1.plot(f_thz, y_d1_noisy, 'g--', lw=2, label='With Noise (Amp+Phase)')
    ax1.set_title('Day 1: Baseline State')
    ax1.set_xlabel('Frequency (THz)')
    ax1.set_ylabel('Transmission (dB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    y_min, y_max = np.min(y_d1), np.max(y_d1)
    ax1.set_ylim([y_min - 5, y_max + 5])  # 留出5dB的边距

    # 子图2-4: 各种胁迫场景
    for idx, (wc_s, d_s, off_d, noise_lvl, label, color) in enumerate(scenarios):
        ax = axes.flat[idx + 1]

        # 生成胁迫参数
        p_stress = apply_water_stress(p_day1, wc_s, d_s, off_d)

        # 理想曲线
        y_clean = forward_solver(p_stress, f_hz)

        # 带噪声曲线 (包含相位噪声)
        y_noisy = NoiseSimulator.add_realistic_noise(y_clean, f_hz, p_stress,
                                                     noise_lvl, include_phase=True)

        # 绘图
        ax.scatter(f_thz, y_d4, c='red', marker='x', alpha=0.5, s=30, label='Day 4 Exp.')
        # ax.plot(f_thz, y_clean, color=color, lw=2, label='Ideal Simulation')
        ax.plot(f_thz, y_noisy, color=color, ls='--', lw=2, label='With Noise')

        # 计算误差
        mse_clean = np.mean((y_d4 - y_clean) ** 2)
        mse_noisy = np.mean((y_d4 - y_noisy) ** 2)

        ax.set_title(f'{label}\nWC={wc_s:.0%}, d={d_s:.0%}\n'
                     f'MSE: Ideal={mse_clean:.2f}, Noisy={mse_noisy:.2f}')
        ax.set_xlabel('Frequency (THz)')
        ax.set_ylabel('Transmission (dB)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        y_min, y_max = np.min(y_d4), np.max(y_d4)
        ax.set_ylim([y_min - 5, y_max + 5])  # 留出5dB的边距

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("模拟完成!")
    print("\n>>> 仿真流程总结:")
    print("  1. 加载实验数据 (Day 1 和 Day 4)")
    print("  2. 使用转移矩阵法拟合 Day 1 基准参数")
    print("  3. 定义水分胁迫转换函数 (WC缩放 + 厚度缩放)")
    print("  4. 应用真实噪声模型生成模拟数据")
    print("  5. 可视化对比理想/带噪声/实验结果")

    print("\n>>> 噪声模型总结:")
    print("\n  【振幅噪声】(5种)")
    print("    ├─ 探测器噪声: 高斯白噪声, SNR 25-40 dB")
    print("    ├─ 频率相关噪声: 高频衰减, 幅度 0.15-0.5 dB")
    print("    ├─ 系统漂移: 低频正弦基线漂移, 幅度 0.1-0.3 dB")
    print("    ├─ 法布里-珀罗效应: 周期性波纹, 幅度 0.1-0.25 dB")
    print("    └─ 定位误差: 光程差变化, 幅度 0.03-0.08 dB")

    print("\n  【相位噪声】(3种)")
    print("    ├─ 湍流噪声: 白相位噪声, 标准差 0.02-0.10 rad")
    print("    ├─ 色散噪声: 频率相关相位误差, 因子 0.01-0.04")
    print("    └─ 相关噪声: 低频平滑漂移, 幅度 0.015-0.05 rad")

    print("\n  【噪声等级】")
    print("    • low    - 理想实验室条件")
    print("    • medium - 典型测量环境 (推荐)")
    print("    • high   - 恶劣环境/快速测量")

    print("\n  【物理机制】")
    print("    振幅噪声 → 透射强度波动")
    print("    相位噪声 → 干涉峰位漂移 + 谱线展宽")
    print("=" * 70)


if __name__ == "__main__":
    main()