# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def preprocess_data(file1: Path, file2: Path) -> pd.DataFrame:
    df1 = pd.read_csv(file1)
    #df2 = pd.read_csv(file2)

    df = df1

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['delay_ms'] = pd.to_numeric(df['delay_ms'], errors='coerce')
    df['packet_loss'] = (df['delay_ms'] == -1).astype(int)
    df['pure_delay'] = df['delay_ms'].replace(-1, pd.NA)
    df['delay_ms'] = df['delay_ms'].replace(-1, 999)

    delay_for_smooth = df['delay_ms'].mask(df['delay_ms'] == 999)
    df['smoothed_delay'] = (
        delay_for_smooth
        .rolling(window=1000, min_periods=1, center=True)
        .mean()
    )

    return df

def plot_fiber_delay_and_loss(df, title='CPE A → B (Fiber Only)'):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df['time'], df['smoothed_delay'], label='Fiber Delay (Smoothed)', color='royalblue', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Delay (ms)')
    ax1.set_ylim(0, 30)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.2)

    loss_times = df[df['packet_loss'] == 1]['time']
    for t in loss_times:
        ax1.axvline(x=t, color='red', alpha=0.2, linewidth=1, linestyle='-')  # 可调整 alpha 和 linewidth

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='Packet Loss')
    ax1.legend(handles=[ax1.lines[0], red_patch], loc='upper left', fontsize=10)

    plt.title(f'Packet Loss Event Classification\n{title}', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_mobile_delay_and_loss(df, title='CPE A → B (4G Only)'):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df['time'], df['smoothed_delay'], label='4G Delay (Smoothed)', color='darkorange', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Delay (ms)')
    ax1.set_ylim(0, 300)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax1.grid(True, linestyle='--', alpha=0.2)

    loss_times = df[df['packet_loss'] == 1]['time']
    for t in loss_times:
        ax1.axvline(x=t, color='red', alpha=0.2, linewidth=1, linestyle='-')  # 可调整 alpha 和 linewidth

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', alpha=0.2, label='Packet Loss')
    ax1.legend(handles=[ax1.lines[0], red_patch], loc='upper left', fontsize=10)

    plt.title(f'Packet Loss Event Classification\n{title}', fontsize=13)
    plt.tight_layout()
    plt.show()

def auto_match_and_plot(folder1, folder2, src='cpe_a', dst='cpe_b'):
    fiber_file1 = Path(folder1) / f"{src}-{dst}-fiber.csv"
    fiber_file2 = Path(folder2) / f"{src}-{dst}-fiber.csv"
    mobile_file1 = Path(folder1) / f"{src}-{dst}-mobile.csv"
    mobile_file2 = Path(folder2) / f"{src}-{dst}-mobile.csv"

    df_fiber = preprocess_data(fiber_file1, fiber_file2)
    df_mobile = preprocess_data(mobile_file1, mobile_file2)

    if df_fiber.empty or df_mobile.empty:
        print(f"Empty data for {src} → {dst}, skipping...")
        return

    plot_fiber_delay_and_loss(df_fiber, title=f"{src.upper()} → {dst.upper()} (Fiber Only)")
    plot_mobile_delay_and_loss(df_mobile, title=f"{src.upper()} → {dst.upper()} (4G Only)")

def auto_plot_all_pairs(folder1, folder2):
    nodes = ['cpe_a', 'cpe_b', 'cpe_c']
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            try:
                print(f"Plotting {src} → {dst}...")
                auto_match_and_plot(folder1, folder2, src, dst)
            except FileNotFoundError:
                print(f"Skipped missing pair: {src}-{dst}")
            except Exception as e:
                print(f"Error plotting {src}-{dst}: {e}")

auto_plot_all_pairs(
    "example/1st_capture",
    "example/2nd_capture"
)