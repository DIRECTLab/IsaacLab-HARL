"""
Script to benchmark FPS for different environments with varying number of environments.
Tests each environment with 100, 1000, and 5000 environments.
Saves results to CSV and creates a plot.

Uses subprocess to run train.py in separate processes and parses FPS from console output.
This avoids the multi-simulation context limitation of isaaclab.
"""

import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from isaaclab.utils import HF_POLICY_MAP


def get_test_environments():
    """Get list of test environments from HF_POLICY_MAP."""
    return list(HF_POLICY_MAP.keys())


def benchmark_environment(env_name: str, num_envs: int, algorithm: str = "happo", 
                         num_steps: int = 500) -> float:
    """
    Benchmark an environment by running train.py in a subprocess and parsing FPS output.
    
    Args:
        env_name: Name of the environment to test
        num_envs: Number of parallel environments
        algorithm: Algorithm to use (happo, happo_adv, etc.)
        num_steps: Number of environment steps to run
        
    Returns:
        FPS (frames per second) or None if benchmark failed
    """
    try:
        print(f"  Testing {env_name} with {num_envs} envs (algo: {algorithm})...", end="", flush=True)
        
        # Get the workspace and train script path
        workspace_root = Path(__file__).parent.parent.parent
        train_script = workspace_root / "scripts" / "reinforcement_learning" / "harl" / "train.py"

        num_steps = num_envs * num_steps  # Total steps across all envs
        
        # Build command with unbuffered Python output
        cmd = [
            sys.executable,
            "-u",  # Unbuffered output
            str(train_script),
            "--algorithm", algorithm,
            "--num_envs", str(num_envs),
            "--num_env_steps", str(num_steps),
            "--task", env_name,
            "--save_interval", "1000000",  # Don't save during benchmark
            "--log_interval", "1",
            "--headless",
        ]
        
        # Run training subprocess and capture both stdout and stderr
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(workspace_root),
            bufsize=1  # Line buffering
        )
        
        last_fps = None
        
        # Capture complete output from process
        try:
            stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
            
            # Combine stdout and stderr for parsing
            combined_output = stdout + "\n" + stderr
            
            # Search for all FPS matches in output
            fps_matches = re.findall(r'FPS\s+(\d+\.?\d*)', combined_output)
            if fps_matches:
                last_fps = float(fps_matches[-1])  # Use the last FPS found
        except subprocess.TimeoutExpired:
            print(f" ERROR: Timeout (>1 hour)")
            process.kill()
            try:
                stdout, stderr = process.communicate(timeout=10)
                if stdout or stderr:
                    combined_output = stdout + "\n" + stderr
                    fps_matches = re.findall(r'FPS\s+(\d+\.?\d*)', combined_output)
                    if fps_matches:
                        last_fps = float(fps_matches[-1])
            except:
                pass
            return None
        
        # Use the last FPS value found
        if last_fps is not None:
            print(f" FPS: {last_fps:.2f}")
            return last_fps
        else:
            print(f" ERROR: No FPS found in output")
            return None
        
    except subprocess.TimeoutExpired:
        print(f" ERROR: Timeout (>1 hour)")
        process.kill()
        return None
    except Exception as e:
        print(f" ERROR: {str(e)}")
        return None


def main():
    # Configuration
    test_envs = get_test_environments()
    num_envs_list = [100, 1000, 5000]
    num_steps = 500  # Steps per benchmark (enough to get stable FPS)
    
    # Results storage
    results = []
    
    print("=" * 80)
    print("FPS Benchmark Suite (Via Subprocess Training)")
    print("=" * 80)
    print(f"Environments to test: {len(test_envs)}")
    print(f"Num envs configurations: {num_envs_list}")
    print(f"Steps per benchmark: {num_steps}")
    print("=" * 80)
    print()
    
    # Run benchmarks
    try:
        for i, env_name in enumerate(test_envs, 1):
            print(f"[{i}/{len(test_envs)}] Environment: {env_name}")
            
            # Get algorithm for this environment
            algorithm = HF_POLICY_MAP[env_name].get("algorithm", "happo")
            
            for num_envs in num_envs_list:
                fps = benchmark_environment(env_name, num_envs, algorithm, num_steps)
                
                if fps is not None:
                    results.append({
                        'environment': env_name,
                        'num_envs': num_envs,
                        'fps': fps,
                        'timestamp': datetime.now().isoformat()
                    })
            
            print()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    
    # Save to CSV
    if results:
        output_dir = Path(__file__).parent / "benchmark_results"
        output_dir.mkdir(exist_ok=True)
        
        csv_file = output_dir / f"fps_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['environment', 'num_envs', 'fps', 'timestamp'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {csv_file}")
        
        # Create plots
        create_plots(results, output_dir)
    else:
        print("No successful benchmarks to save.")


def create_plots(results: list, output_dir: Path):
    """Create visualizations of benchmark results."""
    
    # Organize data for plotting
    env_names = sorted(set(r['environment'] for r in results))
    num_envs_list = sorted(set(r['num_envs'] for r in results))
    
    # Plot 1: Line plot - FPS by num_envs for each environment
    fig, ax = plt.subplots(figsize=(12, 6))
    for env_name in env_names:
        env_results = [r for r in results if r['environment'] == env_name]
        env_results.sort(key=lambda x: x['num_envs'])
        
        if env_results:
            x = [r['num_envs'] for r in env_results]
            y = [r['fps'] for r in env_results]
            ax.plot(x, y, marker='o', label=env_name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
    ax.set_ylabel('FPS', fontsize=11, fontweight='bold')
    ax.set_title('FPS vs Number of Environments', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    
    line_plot_file = output_dir / f"fps_line_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(line_plot_file, dpi=150, bbox_inches='tight')
    print(f"Line plot saved to: {line_plot_file}")
    plt.close(fig)
    
    # Plot 2: Bar chart for each num_envs configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(env_names))
    width = 0.25
    
    for i, num_envs in enumerate(num_envs_list):
        fps_values = []
        for env_name in env_names:
            env_results = [r for r in results if r['environment'] == env_name and r['num_envs'] == num_envs]
            fps = env_results[0]['fps'] if env_results else 0
            fps_values.append(fps)
        
        ax.bar(x + i*width, fps_values, width, label=f'{num_envs} envs')
    
    ax.set_xlabel('Environment', fontsize=11, fontweight='bold')
    ax.set_ylabel('FPS', fontsize=11, fontweight='bold')
    ax.set_title('FPS Comparison by Number of Environments', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in env_names], 
                        rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    bar_plot_file = output_dir / f"fps_bars_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(bar_plot_file, dpi=150, bbox_inches='tight')
    print(f"Bar plot saved to: {bar_plot_file}")
    plt.close(fig)
    
    # Plot 3: Heatmap of FPS values
    fig, ax = plt.subplots(figsize=(10, 8))
    fps_matrix = np.zeros((len(env_names), len(num_envs_list)))
    
    for i, env_name in enumerate(env_names):
        for j, num_envs in enumerate(num_envs_list):
            env_results = [r for r in results if r['environment'] == env_name and r['num_envs'] == num_envs]
            fps_matrix[i, j] = env_results[0]['fps'] if env_results else 0
    
    im = ax.imshow(fps_matrix, cmap='YlGn', aspect='auto')
    ax.set_xticks(np.arange(len(num_envs_list)))
    ax.set_yticks(np.arange(len(env_names)))
    ax.set_xticklabels(num_envs_list)
    ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in env_names], fontsize=9)
    ax.set_xlabel('Number of Environments', fontsize=11, fontweight='bold')
    ax.set_ylabel('Environment', fontsize=11, fontweight='bold')
    ax.set_title('FPS Heatmap', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('FPS', fontsize=10, fontweight='bold')
    
    # Add text annotations
    for i in range(len(env_names)):
        for j in range(len(num_envs_list)):
            fps_val = fps_matrix[i, j]
            if fps_val > 0:
                ax.text(j, i, f'{fps_val:.0f}', ha="center", va="center", 
                        color="black" if fps_val < fps_matrix.max()/2 else "white", fontsize=8)
    
    plt.tight_layout()
    heatmap_plot_file = output_dir / f"fps_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(heatmap_plot_file, dpi=150, bbox_inches='tight')
    print(f"Heatmap plot saved to: {heatmap_plot_file}")
    plt.close(fig)
    
    # Plot 4: Summary statistics
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    summary_text = "Summary Statistics:\n\n"
    summary_text += f"Total Benchmarks: {len(results)}\n"
    summary_text += f"Environments Tested: {len(env_names)}\n"
    summary_text += f"Configurations: {num_envs_list}\n\n"
    
    summary_text += "FPS Statistics:\n"
    all_fps = [r['fps'] for r in results if r['fps']]
    if all_fps:
        summary_text += f"  Mean FPS: {np.mean(all_fps):.2f}\n"
        summary_text += f"  Min FPS: {np.min(all_fps):.2f}\n"
        summary_text += f"  Max FPS: {np.max(all_fps):.2f}\n"
        summary_text += f"  Std Dev: {np.std(all_fps):.2f}\n\n"
    
    summary_text += "Top 5 Configurations:\n"
    top_results = sorted(results, key=lambda x: x['fps'], reverse=True)[:5]
    for i, r in enumerate(top_results, 1):
        short_name = r['environment'][:25] + '...' if len(r['environment']) > 25 else r['environment']
        summary_text += f"  {i}. {short_name}\n"
        summary_text += f"     {r['num_envs']} envs: {r['fps']:.2f} FPS\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    summary_plot_file = output_dir / f"fps_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_plot_file}")
    plt.close(fig)


if __name__ == "__main__":
    main()
