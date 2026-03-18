import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
import os

# ================= 0. Physical Constants & Experimental Setup =================
c = 2.99792458e8        # Speed of light (m/s)
EPS_0 = 8.854187817e-12 # Vacuum permittivity (F/m)

# Fixture Parameters: PTFE (Teflon) is usually used as a leaf holder.
# PTFE is nearly transparent in the THz band with a stable refractive index.
N_PTFE_REAL = 1.44      
N_PTFE_LOSS = 0.0005    

# Detector Dynamic Range Limit
# manual_floor_db: The system's noise floor wall. When the signal attenuates 
# beyond this value, we cannot distinguish the signal and only read noise.
manual_floor_db = 100   # Unit: dB (Signal is clipped if 100dB below transmission power)


# ================= 1. Microscopic Dielectric Kernel =================

def get_leaf_refractive_index(wc, f_hz, sigma=0.1, ph=7.0, temp=25.0):
    """
    [Physical Level: Microscopic]
    Calculates the complex refractive index of the leaf. The leaf is not a single 
    medium but a mixture of water, dry matter, and ions.
    
    Parameters:
        wc: Water Content (0~1)
        f_hz: Frequency (Hz)
        sigma: Conductivity (S/m) -> Reflects salinity stress level
        ph: Acidity/Alkalinity -> Affects water molecule relaxation properties
        temp: Temperature (C) -> Affects the dielectric constant of water
    """
    
    # --- 1.1 Temperature Correction for Water ---
    # Physical Phenomenon: As temperature rises, thermal motion of water molecules increases,
    # causing the static dielectric constant (epsilon_s) to decrease.
    # Empirical formula: epsilon_s drops by ~0.35 for every 1 degree rise.
    eps_s_base = 78.4 - 0.35 * (temp - 25.0)

    # --- 1.2 pH Binding Effect on Water ---
    # Physical Phenomenon: When pH deviates from neutral, ion concentration changes, 
    # binding water molecules. This lengthens relaxation time (tau) and lowers 
    # the effective dielectric constant.
    ph_dev = np.abs(ph - 7.0)
    eps_s_mod = eps_s_base * (1 - 0.015 * ph_dev)   # Static permittivity correction
    tau_1_mod = 8.27e-12 * (1 + 0.01 * ph_dev)      # Slow relaxation time correction

    # --- 1.3 Double Debye Water Model ---
    # Describes two main relaxation processes of free water in the THz band:
    # tau_1: Reorientation of water molecules (Slow, ~8ps)
    # tau_2: Stretching vibration of hydrogen bond networks (Fast, ~0.18ps)
    eps_inf, eps_s, eps_1 = 4.9, eps_s_mod, 5.2
    tau_1, tau_2 = tau_1_mod, 0.18e-12
    omega = 2 * np.pi * f_hz

    # Complex permittivity of pure water
    eps_w = eps_inf + \
            (eps_s - eps_1) / (1 - 1j * omega * tau_1) + \
            (eps_1 - eps_inf) / (1 - 1j * omega * tau_2)

    # --- 1.4 Ionic Conductivity Loss (Ohmic Loss) ---
    # Physical Phenomenon: Salts (sigma) generate current under an electric field,
    # converting electromagnetic energy into heat.
    # This mainly affects the imaginary part (loss component) of the permittivity.
    eps_sigma = 1j * sigma / (omega * EPS_0)
    eps_w_total = eps_w + eps_sigma

    # --- 1.5 Landau-Lifshitz-Looyenga (LLL) Mixing Theory ---
    # Mixes "Water" and "Dry Matter" (Cellulose, etc.) by volume ratio to calculate
    # the effective permittivity of the whole leaf.
    # eps_dry: Permittivity of dry matter (Real part ~2.5, imaginary part very small)
    eps_dry = complex(2.5, 0.05)
    
    # Mixing formula: eps_eff^(1/3) = wc * eps_w^(1/3) + (1-wc) * eps_dry^(1/3)
    term_w = wc * (eps_w_total ** (1 / 3))
    term_d = (1 - wc) * (eps_dry ** (1 / 3))
    eps_eff = (term_w + term_d) ** 3

    # --- 1.6 Convert to Complex Refractive Index (n - ik) ---
    # Wave equations usually use refractive index n (velocity) and extinction coefficient k (attenuation)
    n_complex = np.sqrt(eps_eff)
    n_real = np.real(n_complex)
    n_imag = np.abs(np.imag(n_complex)) # Ensure the loss term is positive

    return complex(n_real, -n_imag)


# ================= 2. Macroscopic Wave Propagation =================

def transfer_matrix_solver(f_hz, d_leaf_mm, wc, d_ptfe_mm, sigma, ph, phase_noise=0.0, return_complex=False, temp=25.0):
    """
    [Physical Level: Macroscopic]
    Uses the Transfer Matrix Method (TMM) to calculate wave propagation through a multilayer structure.
    Structure Model: Air -> PTFE Fixture -> Leaf -> PTFE Fixture -> Air
    
    This function naturally includes Fabry-Pérot interference effects 
    (i.e., the oscillating ripples seen in the THz spectrum).
    """
    k0 = 2 * np.pi * f_hz / c # Vacuum wavenumber
    
    # Define refractive indices for layers
    n_air = 1.0 + 0j
    n_ptfe = complex(N_PTFE_REAL, -N_PTFE_LOSS)
    # Get leaf refractive index based on current state (Calls microscopic kernel above)
    n_leaf = get_leaf_refractive_index(wc, f_hz, sigma, ph, temp=temp)

    d_ptfe = d_ptfe_mm * 1e-3
    d_leaf = d_leaf_mm * 1e-3

    # --- 2.1 Interface Matrix ---
    # Describes reflection (r) and transmission (t) at the interface of two media - Fresnel Equations
    def interface(n1, n2):
        r = (n1 - n2) / (n1 + n2)
        t = 2 * n1 / (n1 + n2)
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    # --- 2.2 Propagation Matrix ---
    # Describes phase accumulation (-1j*n*k0*d) and absorption loss inside a medium
    def prop(n, d, add_phase_noise=False):
        phase = -1j * n * k0 * d
        # Numerical stability handling: Prevent exponential explosion
        phase = np.clip(phase.real, -50, 50) + 1j * phase.imag
        
        # Inject phase noise (Simulating wavefront distortion due to medium inhomogeneity)
        if add_phase_noise and phase_noise != 0.0:
            phase += 1j * phase_noise
            
        p = np.exp(phase)
        return np.array([[1 / p, 0], [0, p]], dtype=complex)

    # --- 2.3 Matrix Cascading (Multiplication) ---
    # Multiply in optical order: Air->PTFE->Leaf->PTFE->Air
    M = interface(n_air, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_leaf) @ prop(n_leaf, d_leaf, add_phase_noise=True) @ \
        interface(n_leaf, n_ptfe) @ prop(n_ptfe, d_ptfe) @ \
        interface(n_ptfe, n_air)

    # Transmission coefficient t = 1 / M[0,0]
    t_complex = 1.0 / M[0, 0]

    if return_complex:
        return t_complex

    # Power transmission T = |t|^2
    T = np.abs(t_complex) ** 2
    return np.clip(T, 1e-15, 1.0)


# ================= 3. Environmental & System Coupling =================

def forward_solver(params, f_hz, phase_noise_array=None, return_complex=False, temp=25.0, rh=50.0):
    """
    [Physical Level: System & Environment]
    Places macroscopic electromagnetic results into a real environment, 
    considering atmospheric attenuation, scattering, and detector noise floor.
    
    params: [wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff]
    """
    # Parameter unpacking and padding
    p = list(params)
    while len(p) < 7:
        if len(p) == 4: p.append(0.1)
        if len(p) == 5: p.append(7.0)
        if len(p) == 6: p.append(0.0)
    wc, d_leaf, d_ptfe, offset, sigma, ph, scat_coeff = p

    # --- 3.1 Atmospheric Path Loss ---
    # Physical Phenomenon: Water vapor in the air absorbs THz waves. Higher humidity (RH) means stronger absorption.
    # Empirical Model: 0.002 * RH * f^2 (Simple approximation model)
    f_thz = f_hz / 1e12
    humidity_loss_db = 0.002 * rh * (f_thz**2)

    if phase_noise_array is None:
        phase_noise_array = np.zeros(len(f_hz))

    # --- 3.2 Call TMM to calculate base response ---
    T_list = np.array([
        transfer_matrix_solver(f, d_leaf, wc, d_ptfe, sigma, ph, pn, return_complex=True, temp=temp)
        for f, pn in zip(f_hz, phase_noise_array)
    ])

    # --- 3.3 Physical Loss Coupling ---
    # Apply Environmental Loss (Humidity) and Structural Scattering (Mie/Rayleigh) to the complex amplitude.
    # Scattering loss is usually proportional to the square of frequency (f^2).
    total_attenuation_db = humidity_loss_db + (scat_coeff * (f_thz ** 2))
    attenuation_factor = 10**(-total_attenuation_db / 20) # Convert to linear amplitude ratio
    T_final = T_list * attenuation_factor

    # --- 3.4 Detector Noise Floor Clipping Effect ---
    # Physical Phenomenon: When the signal is too weak, it gets buried in the detector's background noise.
    # This causes the signal waveform to "flatten" at the bottom (clipping/paralysis). 
    # This has a huge impact on feature extraction.
    
    floor_linear = 10 ** (-manual_floor_db / 10) # Linear power threshold for noise floor

    # Calculate magnitude
    mag = np.abs(T_final)
    # Force clipping: Magnitude cannot be lower than the square root of the noise floor
    mag_clipped = np.maximum(mag, np.sqrt(floor_linear))

    # Reconstruct complex signal: Keep original phase, but amplitude is corrected (raised by noise floor)
    T_final_clipped = mag_clipped * np.exp(1j * np.angle(T_final))

    if return_complex:
        return T_final_clipped

    # Convert to dB and add a linear offset at the end
    y_db = 10 * np.log10(mag_clipped ** 2) + offset
    return y_db


# ================= 4. Hardware Noise Simulation =================

class NoiseSimulator:
    """
    [Physical Level: Hardware Reality]
    Simulates various errors introduced by non-ideal hardware.
    """

    @staticmethod
    def detector_noise(signal, snr_db=30):
        # Thermal Noise / Shot Noise: Additive white Gaussian noise
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        return noise

    @staticmethod
    def frequency_dependent_noise(f_hz, amplitude=0.3):
        # Frequency Dependent Noise: SNR usually worsens at high frequencies due to lower transmission power
        f_thz = f_hz / 1e12
        noise_profile = amplitude * (1 + 0.5 * (f_thz - 0.75))
        return np.random.normal(0, noise_profile)

    @staticmethod
    def systematic_drift(f_hz, drift_amplitude=0.2):
        # Systematic Drift: Low-frequency baseline sway caused by device warm-up or environmental temp fluctuations
        f_thz = f_hz / 1e12
        drift = drift_amplitude * np.sin(2 * np.pi * (f_thz - 0.75) / 0.35)
        return drift

    @staticmethod
    def etalon_effect(f_hz, period_thz=0.05, amplitude=0.15):
        # Etalon Effect: Parasitic interference fringes caused by parallel surfaces like lenses or emitter windows
        f_thz = f_hz / 1e12
        ripple = amplitude * np.sin(2 * np.pi * f_thz / period_thz)
        return ripple

    @staticmethod
    def sample_positioning_error(signal, position_std=0.05):
        # Positioning Error: Random phase jitter caused by micron-level differences in sample placement
        phase_noise = np.random.normal(0, position_std, len(signal))
        return phase_noise

    # --- Phase Noise Generators (Used to simulate frequency domain jitter) ---
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
        # Combine the above phase noises
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
        """
        Comprehensive Noise Injection: Superimpose all types of noise onto the ideal signal
        """
        amp_noise_params = {
            'low': {'snr': 40, 'freq': 0.15, 'drift': 0.1, 'etalon': 0.1, 'pos': 0.03},
            'medium': {'snr': 30, 'freq': 0.30, 'drift': 0.2, 'etalon': 0.15, 'pos': 0.05},
            'high': {'snr': 25, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08},
            'reallyH': {'snr': 5, 'freq': 0.70, 'drift': 0.5, 'etalon': 0.5, 'pos': 0.2}
        }
        p = amp_noise_params[noise_level]

        if include_phase:
            phase_noise = NoiseSimulator.generate_phase_noise(f_hz, noise_level)
            # Re-calculate the complex signal with phase noise using forward_solver
            noisy_signal = forward_solver(params, f_hz, phase_noise)
        else:
            noisy_signal = signal.copy()

        # Add amplitude-based noises (dB domain)
        noisy_signal += NoiseSimulator.detector_noise(signal, p['snr'])
        noisy_signal += NoiseSimulator.frequency_dependent_noise(f_hz, p['freq'])
        noisy_signal += NoiseSimulator.systematic_drift(f_hz, p['drift'])
        noisy_signal += NoiseSimulator.etalon_effect(f_hz, amplitude=p['etalon'])
        noisy_signal += NoiseSimulator.sample_positioning_error(signal, p['pos'])
        return noisy_signal


# ================= 5. Biological to Physical Mapping =================

def apply_comprehensive_stress(p_d1, wc_s, d_s, sigma_s, ph_v, scat_v, off_d):
    """
    Maps biological stress states to changes in physical parameters.
    Examples:
    - Drought -> wc_s (water content) decreases, d_s (thickness) decreases
    - Salinity -> sigma_s (conductivity) increases drastically
    """
    return [
        p_d1[0] * wc_s,   # Water (Multiplier factor)
        p_d1[1] * d_s,    # Thickness (Multiplier factor)
        p_d1[2],          # PTFE Thickness (Unchanged)
        p_d1[3] + off_d,  # Offset Drift (Addition)
        p_d1[4] * sigma_s,# EC (Multiplier factor)
        ph_v,             # pH (Direct setting)
        scat_v            # Scattering coefficient (Direct setting)
    ]


# ================= 6. Signal Processing & Feature Extraction =================

def extract_paper_features(f_hz, T_complex, reference_T_complex=None):
    """
    Fully replicates the feature extraction method from the paper.
    Includes Frequency Domain Statistical Features and Time Domain Waveform Features.
    """
    features = {}

    # --- 6.1 Frequency Domain Features ---
    # Calculate power spectrum, split into 5 bins, calculate mean and max for each bin
    power_spectrum = np.abs(T_complex) ** 2
    bins = np.array_split(power_spectrum, 5)

    for i, b in enumerate(bins):
        features[f'Bin{i + 1}_Var'] = np.mean(b)  # Bin Average Energy
        features[f'Bin{i + 1}_Max'] = np.max(b)   # Bin Peak Energy

    # --- 6.2 Time Domain Features ---
    # Use IFFT (Inverse Fast Fourier Transform) to convert frequency data back to time domain pulse
    pulse = np.abs(np.fft.irfft(T_complex))

    # Extract statistical moments of the time domain waveform
    features['TD_Std'] = np.std(pulse)        # Standard Deviation
    features['TD_Skew'] = skew(pulse)         # Skewness (Measure of asymmetry)
    features['TD_Kurtosis'] = kurtosis(pulse) # Kurtosis (Measure of "tailedness")
    features['TD_Median'] = np.median(pulse)
    features['TD_AbsDev'] = np.mean(np.abs(pulse - np.mean(pulse))) # Absolute Deviation

    q1, q3 = np.percentile(pulse, [25, 75])
    features['TD_Q1'] = q1
    features['TD_Q3'] = q3
    features['TD_IQR'] = q3 - q1

    # PCC (Pearson Correlation Coefficient): Measures similarity between current signal and ideal air reference
    # Higher stress -> Higher signal distortion -> Lower PCC
    if reference_T_complex is not None:
        ref_pulse = np.abs(np.fft.irfft(reference_T_complex))
        min_len = min(len(pulse), len(ref_pulse))
        features['TD_PCC'] = np.corrcoef(pulse[:min_len], ref_pulse[:min_len])[0, 1]
    else:
        features['TD_PCC'] = 1.0

    return features


# ================= 7. Data Generation =================

def generate_raw_dataset(p_day1, n_samples_per_class=50, noise_level='medium', temp=25.0, rh=50.0):
    """
    Main function for generating simulated experimental data.
    It iterates through different physiological scenarios, calls the physics model 
    to generate data, and adds noise.
    """

    # Scenario Configuration List: 
    # (Water Factor, Thickness Factor, Conductivity Factor, pH, Scattering, Offset, Temp, Humidity, Noise Level, Label)
    configs = [
        # --- Standard Control Group ---
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'Normal'),
        
        # --- Drought Stress (Reduced water, thinner leaf) ---
        (0.75, 0.80, 1.0, 7.0, 0.0, 0.0, 25.0, 50.0, 'high', 'A_Drought'),
        
        # --- Environmental Interference Control (High Humidity) ---
        # Used to test if the algorithm misclassifies simple environmental humidity attenuation as leaf drought
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 95.0, 'medium', 'F_High_Humidity'),
        (1.0, 1.0, 1.0, 7.0, 0.0, 0.0, 25.0, 5.0, 'medium', 'G_Low_Humidity'),
    ]

    f_hz = np.linspace(0.80, 1.07, 100) * 1e12 # Define frequency range
    
    # Define specific parameters for different noise levels
    amp_noise_params = {
        'low': {'snr': 40, 'freq': 0.15, 'drift': 0.1, 'etalon': 0.1, 'pos': 0.03},
        'medium': {'snr': 30, 'freq': 0.30, 'drift': 0.2, 'etalon': 0.15, 'pos': 0.05},
        'high': {'snr': 15, 'freq': 0.50, 'drift': 0.3, 'etalon': 0.25, 'pos': 0.08},
        'reallyH': {'snr': 5, 'freq': 0.1, 'drift': 0.5, 'etalon': 0.5, 'pos': 0.2}
    }

    all_data = []

    for wc_s, d_s, sig_s, ph_v, scat_v, off_d, t_s, r_s, nl_s, label in configs:

        # 1. Generate "Air Reference Signal" for the current environment
        # In experiments, an air measurement is usually taken before each measurement as a baseline.
        air_params = [0.0, 0.0, p_day1[2], p_day1[3], 0.0, 7.0, 0.0]
        reference_T_complex = forward_solver(air_params, f_hz, return_complex=True, temp=t_s, rh=r_s)

        # 2. Get noise parameters
        nz = amp_noise_params[nl_s]
        print(f"Generating Scenario: {label} (Env:{t_s}C/{r_s}%RH, Noise:{nl_s})")

        # Calculate center physical parameters for this scenario
        p_center = apply_comprehensive_stress(p_day1, wc_s, d_s, sig_s, ph_v, scat_v, off_d)

        # 3. Loop to generate samples (Monte Carlo Simulation)
        for i in range(n_samples_per_class):
            # Apply slight random perturbations to physical parameters (Simulating individual variability)
            wc_i = np.clip(p_center[0] + np.random.normal(0, 0.015), 0.1, 0.98)
            d_i = np.clip(p_center[1] + np.random.normal(0, 0.02), 0.05, 0.8)
            sig_i = np.clip(p_center[4] + np.random.normal(0, p_center[4] * 0.05), 0.01, 100)
            ph_i = np.clip(p_center[5] + np.random.normal(0, 0.1), 2.0, 9.0)
            scat_i = np.clip(p_center[6] + np.random.normal(0, 1.0), 0, 50)
            off_i = p_center[3] + np.random.normal(0, 0.3)
            current_p = [wc_i, d_i, p_day1[2], off_i, sig_i, ph_i, scat_i]

            # A. Calculate Ideal Physical Response (Including environmental attenuation)
            T_ideal = forward_solver(current_p, f_hz, return_complex=True, temp=t_s, rh=r_s)

            # B. Inject Complex Noise
            # B1. Phase Noise
            pn = NoiseSimulator.generate_phase_noise(f_hz, noise_level=nl_s)
            T_noisy = T_ideal * np.exp(1j * pn)

            # B2. Amplitude-related Noise (Drift, Ripple, etc.)
            mag_noise_db = (NoiseSimulator.frequency_dependent_noise(f_hz, nz['freq']) +
                            NoiseSimulator.systematic_drift(f_hz, nz['drift']) +
                            NoiseSimulator.etalon_effect(f_hz, amplitude=nz['etalon']))
            T_noisy *= 10 ** (mag_noise_db / 20)

            # B3. Thermal Noise (SNR)
            snr_linear = 10 ** (nz['snr'] / 10)
            noise_std = np.mean(np.abs(T_ideal)) / np.sqrt(snr_linear)
            T_noisy += (np.random.normal(0, noise_std, len(f_hz)) + 1j * np.random.normal(0, noise_std, len(f_hz)))

            # C. Extract Features
            paper_features = extract_paper_features(f_hz, T_noisy, reference_T_complex)

            # D. Save Data
            sample_record = {
                'label': label,
                **paper_features,
            }
            all_data.append(sample_record)

    return pd.DataFrame(all_data)


# ================= 8. Main Execution =================

if __name__ == "__main__":
    # 1. Set Baseline Physical Parameters (Simulating healthy Day 1 Spinach)
    # [WC, d_leaf(mm), d_ptfe(mm), offset(dB), sigma(S/m), pH, Scat]
    fitted_p_day1 = [0.97, 0.5472, 1.867, -2.27, 0.1, 7.0, 0.0]

    # 2. Generate Dataset (100 samples per class)
    df_raw = generate_raw_dataset(fitted_p_day1, n_samples_per_class=100)

    # 3. Save
    output_path = "Spinach_Raw_THz_Stress_Dataset.csv"
    df_raw.to_csv(output_path, index=False)

    print("-" * 50)
    print(f"Data generation complete.")
    print(f"File saved to: {output_path}")
    print(f"Total samples: {df_raw.shape[0]}")