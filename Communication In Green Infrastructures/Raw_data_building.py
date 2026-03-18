#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import os

c = 2.99792458e8
EPS_0 = 8.854187817e-12  # 真空介电常数
N_PTFE_REAL = 1.44
N_PTFE_LOSS = 0.0005
manual_floor_db = 100 # minus

def get_leaf_refractive_index(wc, f_hz, sigma=0.1, ph=7.0, temp=25.0):

    # 静态介电常数随温度升高而降低 (约每度 -0.35)
    eps_s_base = 78.4 - 0.35 * (temp - 25.0)

    # pH 对水分子弛豫特性的物理修正
    ph_dev = np.abs(ph - 7.0)
    eps_s_mod = eps_s_base * (1 - 0.015 * ph_dev)
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
    n_real = np.real(n_complex)
    n_imag = np.abs(np.imag(n_complex))

    return complex(n_real, -n_imag)


def transfer_matrix_solver(f_hz, d_leaf_mm, wc, d_ptfe_mm, sigma, ph, phase_noise=0.0, return_complex=False, temp=25.0):

    k0 = 2 * np.pi * f_hz / c
    n_air = 1.0 + 0j
    n_ptfe = complex(N_PTFE_REAL, -N_PTFE_LOSS)
    n_leaf = get_leaf_refractive_index(wc, f_hz, sigma, ph, temp=temp)

    d_ptfe = d_ptfe_mm * 1e-3
    d_leaf = d_leaf_mm * 1e-3

    def interface(n1, n2):
        r = (n1 - n2) / (n1 + n2)
        t = 2 * n1 / (n1 + n2)
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def prop(n, d, add_phase_noise=False):
        phase = -1j * n * k0 * d
        phase = np.clip(phase.real, -50, 50) + 1j * phase.imag
        if add_phase_noise and phase_noise != 0.0:
            phase += 1j * phase_noise
        p = np.exp(phase)
        return np.array([[1 / p, 0], [0, p]], dtype=complex)

    M = interface(n_air, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_leaf) @ prop(n_leaf, d_leaf, add_phase_noise=True) @ \
        interface(n_leaf, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_air)

    t_complex = 1.0 / M[0, 0]

    if return_complex:
        return t_complex

    T = np.abs(t_complex) ** 2
    return np.clip(T, 1e-15, 1.0)


def forward_solver(params, f_hz, phase_noise_array=None, return_complex=False, temp=25.0, rh=50.0):
    """
    params: [wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff]
    """
    p = list(params)
    while len(p) < 7:
        if len(p) == 4: p.append(0.1)
        if len(p) == 5: p.append(7.0)
        if len(p) == 6: p.append(0.0)
    wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff = p

    # 1. 计算环境湿度带来的空气路径损耗 (dB)
    f_thz = f_hz / 1e12
    humidity_loss_db = 0.002 * rh * (f_thz**2)

    if phase_noise_array is None:
        phase_noise_array = np.zeros(len(f_hz))

    # 2. 核心计算：获取复数响应 T
    T_list = np.array([
        transfer_matrix_solver(f, d_leaf, wc, d_ptfe, sigma, ph, pn, return_complex=True, temp=temp)
        for f, pn in zip(f_hz, phase_noise_array)
    ])

    # 3. 物理损耗耦合：将【湿度衰减】和【结构散射】统一应用在复数幅度上
    # 这样提取出来的 TD_Skew 偏度等特征才能真正反映结构损伤和环境干扰
    total_attenuation_db = humidity_loss_db + (scat_coeff * (f_thz ** 2))
    attenuation_factor = 10**(-total_attenuation_db / 20)
    T_final = T_list * attenuation_factor

    # 底噪墙
    floor_linear = 10 ** (-manual_floor_db / 10)

    # 计算当前信号的模长 (幅度)
    mag = np.abs(T_final)
    # 强制让幅度不低于底噪的开方
    mag_clipped = np.maximum(mag, np.sqrt(floor_linear))

    # 重新构造带截断的复数信号 (保留原有的相位角度)
    T_final_clipped = mag_clipped * np.exp(1j * np.angle(T_final))

    if return_complex:
        return T_final_clipped

    y_db = 10 * np.log10(mag_clipped ** 2) + offset
    return y_db


class NoiseSimulator:

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
            'high': {'turb': 0.10, 'disp': 0.04, 'corr': 0.05},
            'reallyH': {'turb': 0.20, 'disp': 0.10, 'corr': 0.10}
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
            'high': {'snr': 25, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08},
            'reallyH': {'snr': 5, 'freq': 0.70, 'drift': 0.5, 'etalon': 0.5, 'pos': 0.2}
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

def load_data(path):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path, header=None, names=['Freq', 'Db']).groupby('Freq', as_index=False).mean()
    f = np.linspace(0.78, 0.97, 100) * 1e12
    interp = interp1d(df['Freq'] * 1e12, df['Db'], kind='cubic', fill_value="extrapolate")
    return f, interp(f)

from scipy.stats import skew, kurtosis


def extract_paper_features(f_hz, T_complex, reference_T_complex=None):
    features = {}

    # 频域特征: 5个窗口, 每个宽度20
    power_spectrum = np.abs(T_complex) ** 2
    bins = np.array_split(power_spectrum, 5)

    for i, b in enumerate(bins):
        features[f'Bin{i + 1}_Var'] = np.mean(b)
        features[f'Bin{i + 1}_Max'] = np.max(b)

    # 时域特征: IFFT 变换
    pulse = np.abs(np.fft.irfft(T_complex))

    # 9 个统计量
    features['TD_Std'] = np.std(pulse)
    features['TD_Skew'] = skew(pulse)
    features['TD_Kurtosis'] = kurtosis(pulse)
    features['TD_Median'] = np.median(pulse)
    features['TD_AbsDev'] = np.mean(np.abs(pulse - np.mean(pulse)))

    q1, q3 = np.percentile(pulse, [25, 75])
    features['TD_Q1'] = q1  # 25th percentile
    features['TD_Q3'] = q3  # 75th percentile
    features['TD_IQR'] = q3 - q1  # 四分位距

    # PCC (皮尔逊相关系数)
    if reference_T_complex is not None:
        ref_pulse = np.abs(np.fft.irfft(reference_T_complex))
        min_len = min(len(pulse), len(ref_pulse))
        # 计算相关系数矩阵并取 [0,1] 元素
        features['TD_PCC'] = np.corrcoef(pulse[:min_len], ref_pulse[:min_len])[0, 1]
    else:
        features['TD_PCC'] = 1.0

    return features


def generate_raw_dataset(p_day1, n_samples_per_class=50, noise_level='medium', temp=25.0, rh=50.0):

    # (水分, 厚度, 电导率, pH, 散射, 偏移, 温度, 湿度, 噪声等级, 标签)
    configs = [
        # 基础生理场景
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'Normal'),
        (0.75, 0.80, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'A_Drought'),
        #(0.95, 0.95, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'AA_Less_Drought'),
        #(0.95, 0.90, 10.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'low', 'B_Salt'),
        #(0.90, 0.80, 1.0, 7.0, 5.0, 0.0, 25.0, 50.0, 'low', 'C_Structural'),
        #(0.95, 0.90, 1.0, 3.0, 0.0, 0.0, 25.0, 50.0, 'low', 'D_Metabolic'),

        # 植物生长场景
        #(1.0, 1.05, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'E_Growth'),
        # 模拟高湿度环境下信号的衰减是否与干旱容易混淆
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 95.0, 'medium', 'F_High_Humidity'),
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 5.0, 'medium', 'G_Low_Humidity'),
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 45.0, 50.0, 'medium', 'H_High_Temperature'),
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 5.0, 50.0, 'medium', 'I_Low_Temperature'),
    ]

    f_hz = np.linspace(0.80, 1.07, 100) * 1e12
    f_thz = f_hz / 1e12

    # 预定义噪声字典
    amp_noise_params = {
        'low': {'snr': 40, 'freq': 0.15, 'drift': 0.1, 'etalon': 0.1, 'pos': 0.03},
        'medium': {'snr': 30, 'freq': 0.30, 'drift': 0.2, 'etalon': 0.15, 'pos': 0.05},
        'high': {'snr': 15, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08},
        'reallyH': {'snr': 5, 'freq': 0.1, 'drift': 0.5, 'etalon': 0.5, 'pos': 0.2}
    }

    all_data = []

    for wc_s, d_s, sig_s, ph_v, scat_v, off_d, t_s, r_s, nl_s, label in configs:

        # 针对当前场景的环境，准备对应的空气参考信号
        air_params = [0.0, 0.0, p_day1[2], p_day1[3], 0.0, 7.0, 0.0]
        reference_T_complex = forward_solver(air_params, f_hz, return_complex=True, temp=t_s, rh=r_s)

        # 获取当前场景特定的噪声配置
        nz = amp_noise_params[nl_s]

        print(f"场景生成中: {label} (环境:{t_s}C/{r_s}%RH, 噪声:{nl_s})")

        p_center = apply_comprehensive_stress(p_day1, wc_s, d_s, sig_s, ph_v, scat_v, off_d)

        for i in range(n_samples_per_class):
            # 物理参数扰动
            wc_i = np.clip(p_center[0] + np.random.normal(0, 0.015), 0.1, 0.98)
            d_i = np.clip(p_center[1] + np.random.normal(0, 0.02), 0.05, 0.8)
            sig_i = np.clip(p_center[4] + np.random.normal(0, p_center[4] * 0.05), 0.01, 100)
            ph_i = np.clip(p_center[5] + np.random.normal(0, 0.1), 2.0, 9.0)
            scat_i = np.clip(p_center[6] + np.random.normal(0, 1.0), 0, 50)
            off_i = p_center[3] + np.random.normal(0, 0.3)
            current_p = [wc_i, d_i, p_day1[2], off_i, sig_i, ph_i, scat_i]

            # 使用 t_s, r_s 获取理想信号
            T_ideal = forward_solver(current_p, f_hz, return_complex=True, temp=t_s, rh=r_s)

            # 使用 nl_s 注入噪声
            pn = NoiseSimulator.generate_phase_noise(f_hz, noise_level=nl_s)
            T_noisy = T_ideal * np.exp(1j * pn)

            mag_noise_db = (NoiseSimulator.frequency_dependent_noise(f_hz, nz['freq']) +
                            NoiseSimulator.systematic_drift(f_hz, nz['drift']) +
                            NoiseSimulator.etalon_effect(f_hz, amplitude=nz['etalon']))
            T_noisy *= 10 ** (mag_noise_db / 20)

            snr_linear = 10 ** (nz['snr'] / 10)
            noise_std = np.mean(np.abs(T_ideal)) / np.sqrt(snr_linear)
            T_noisy += (np.random.normal(0, noise_std, len(f_hz)) + 1j * np.random.normal(0, noise_std, len(f_hz)))

            # 提取特征
            paper_features = extract_paper_features(f_hz, T_noisy, reference_T_complex)

            # 记录当前样本的真实环境信息
            sample_record = {
                'label': label,
                **paper_features,
            }
            all_data.append(sample_record)

    return pd.DataFrame(all_data)

# ================= 运行示例 =================

def plto_senario():
    # 基础物理参数（拟合出的 Day 1 基准）
    fitted_p_day1 = [0.97, 0.5472, 1.867, -2.27, 0.1, 7.0, 0.0]
    f_hz = np.linspace(0.80, 1.07, 100) * 1e12
    f_thz = f_hz / 1e12

    # 扩展配置：(水分, 厚度, 电导率, pH, 散射, 偏移, 温度, 湿度, 噪声等级, 标签)
    configs = [
        # 基础生理场景
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'Normal'),
        (0.75, 0.65, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'A_Drought'),
        (0.95, 0.95, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'AA_Less_Drought'),
        #(0.95, 0.90, 30.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'medium', 'B_Salt'),
        #(0.70, 0.60, 1.0, 7.0, 15.0, 0.0, 25.0, 50.0, 'medium', 'C_Structural'),
        #(0.95, 0.90, 1.0, 3.0, 0.0, 0.0, 25.0, 50.0, 'medium', 'D_Metabolic'),

        # 植物生长场景
        (1.0, 1.05, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'E_Growth'),

        # 环境干扰场景
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 95.0, 'medium', 'F_High_Humidity'),
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 15.0, 'medium', 'G_Low_Humidity'),
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 45.0, 50.0, 'medium', 'H_High_Temperature'),
        #(1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 5.0, 50.0, 'medium', 'I_Low_Temperature'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes_flat = axes.flatten()

    for idx, (wc_s, d_s, sig_s, ph_v, scat_v, off_d, t_s, r_s, nl_s, label) in enumerate(configs):
        ax = axes_flat[idx]

        # 计算当前场景的物理参数组
        p_s = apply_comprehensive_stress(fitted_p_day1, wc_s, d_s, sig_s, ph_v, scat_v, off_d)

        y_ideal = forward_solver(p_s, f_hz, temp=t_s, rh=r_s)

        # 叠加测量噪声
        y_measured = NoiseSimulator.add_realistic_noise(y_ideal, f_hz, p_s, noise_level=nl_s)

        ax.plot(f_thz, y_measured, label=f'Measured ({nl_s} Noise)')

        ax.set_title(f"Scenario: {label}\nEnv: {t_s}C, {r_s}%RH")
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_ylim([-60, -10])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 这里是拟合时的预设
    fitted_p_day1 = [0.97, 0.5472, 1.867, -2.27, 0.1, 7.0, 0.0]
    df_raw = generate_raw_dataset(fitted_p_day1, n_samples_per_class=100)
    output_path = "Spinach_Raw_THz_Stress_Dataset.csv"
    df_raw.to_csv(output_path, index=False)

    print("-" * 50)
    print(f"原始数据集已生成！保存路径: {output_path}")
    print(f"数据总维度: {df_raw.shape} (每行代表一个叶片样本)")
    print("\n前 5 行预览 (包含物理真值标签和原始 dB 谱):")
    print(df_raw.head())

    #plto_senario()