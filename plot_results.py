import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import argparse

# Set matplotlib font sizes
matplotlib.rcParams.update({
    "font.size": 12,            # Default text size
    "axes.titlesize": 22,       # Title font size
    "axes.labelsize": 22,       # X/Y label font size
    "legend.fontsize": 16,      # Legend font size
    "xtick.labelsize": 18,      # X tick labels
    "ytick.labelsize": 18,      # Y tick labels
})

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results/benchmark_theta_unit_X",
                    help="Directory containing benchmark results")
parser.add_argument("--output_file", type=str, default="results/benchmark_theta_unit_X/mi_comparison_X.pdf",
                    help="Output file for the plot")
parser.add_argument("--estimators", type=str, nargs="+", default=None,
                    help="List of estimators to plot (default: all)")
parser.add_argument("--use_final", action="store_true",
                    help="Use final 1000 steps average instead of full average")

args = parser.parse_args()

def load_results(results_dir):
    """Load all results from the benchmark directory."""
    results = {}
    
    # Find all subdirectories (one per dataset)
    subdirs = sorted([d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d)) and d != "figures"])
    
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        
        # Find all estimator subdirectories
        estimator_dirs = sorted([d for d in os.listdir(subdir_path) 
                                if os.path.isdir(os.path.join(subdir_path, d))])
        
        dataset_results = {}
        
        for estimator_name in estimator_dirs:
            mi_file = os.path.join(subdir_path, estimator_name, "mi.npy")
            if os.path.exists(mi_file):
                mi_estimates = np.load(mi_file)
                dataset_results[estimator_name] = mi_estimates
        
        if dataset_results:
            results[subdir] = dataset_results
    
    return results

def extract_true_mi_from_subdir(subdir_name, summary=None):
    """Extract true MI value from subdirectory name or load from summary."""
    # First try to load from summary file
    if summary is not None:
        key = f'{subdir_name}_true_mi'
        if key in summary:
            return float(summary[key])
    
    # Fallback: try to parse from directory name
    # This is a heuristic - you may need to adjust based on your naming convention
    try:
        # Look for pattern like 0p001, 0p002, etc.
        parts = subdir_name.split('_')
        for part in parts:
            if 'p' in part:
                # Try to extract number like 0p001 -> 0.001
                mi_str = part.replace('p', '.')
                try:
                    return float(mi_str)
                except:
                    pass
    except:
        pass
    
    return None

def plot_results(results, output_file, estimators=None, use_final=False):
    """Plot true MI vs estimated MI for all estimators."""
    
    # Load summary file if available
    summary_file = os.path.join(args.results_dir, "summary.npz")
    summary = None
    if os.path.exists(summary_file):
        summary = np.load(summary_file)
        print(f"Loaded summary file with {len(summary.files)} entries")
    
    # Collect data
    all_true_mi = []
    all_estimates = {}
    
    for subdir_name, dataset_results in sorted(results.items()):
        # Get true MI
        true_mi = extract_true_mi_from_subdir(subdir_name, summary)
        if true_mi is None:
            print(f"Warning: Could not extract true MI for {subdir_name}, skipping...")
            continue
        
        all_true_mi.append(true_mi)
        
        # Get estimates for each estimator
        for estimator_name, estimates in dataset_results.items():
            if estimators is not None and estimator_name not in estimators:
                continue
            
            if estimator_name not in all_estimates:
                all_estimates[estimator_name] = []
            
            if use_final:
                # Use average of last 1000 steps
                estimate = np.mean(estimates[-1000:])
            else:
                # Use average of all steps
                estimate = np.mean(estimates)
            
            all_estimates[estimator_name].append(estimate)
    
    if not all_true_mi:
        print("Error: No valid data found for plotting!")
        return
    
    # Sort by true MI
    sorted_indices = np.argsort(all_true_mi)
    all_true_mi = np.array(all_true_mi)[sorted_indices]
    
    for estimator_name in all_estimates:
        all_estimates[estimator_name] = np.array(all_estimates[estimator_name])[sorted_indices]
    
    # Find global range for both axes (same as combined plot)
    # Calculate min and max across all true MI and all estimates
    all_values = [all_true_mi]
    for estimates in all_estimates.values():
        all_values.append(estimates)
    all_values_flat = np.concatenate(all_values)
    global_min = np.min(all_values_flat)
    global_max = np.max(all_values_flat)
    # Add a small margin (5% on each side)
    margin = (global_max - global_min) * 0.05
    global_min -= margin
    global_max += margin
    
    # Create plot
    n_estimators = len(all_estimates)
    n_cols = 3
    n_rows = (n_estimators + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6.5 * n_rows))

    if n_estimators == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_estimators))
    
    for idx, (estimator_name, estimates) in enumerate(sorted(all_estimates.items())):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(all_true_mi, estimates, alpha=0.6, s=50, color=colors[idx], label=estimator_name)
        # ax.plot(all_true_mi, estimates, linewidth=2.0, marker='o', color=colors[idx], label=estimator_name, alpha=0.6) # Line Plot

        # Perfect estimation line (y=x) - use global range
        ax.plot([global_min, global_max], [global_min, global_max], 'r--', alpha=0.5, label='Perfect estimation')
        
        # Set same axis limits for both x and y (matching combined plot)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect('equal', adjustable='box')  # Make axes equal scale
        
        # Calculate and display correlation
        correlation = np.corrcoef(all_true_mi, estimates)[0, 1]
        rmse = np.sqrt(np.mean((all_true_mi - estimates)**2))
        
        ax.set_xlabel('Ground Truth MI')
        ax.set_ylabel('Estimated MI')
        ax.set_title(f'{estimator_name}\nCorrelation: {correlation:.3f}, RMSE: {rmse:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_estimators, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    # Save as PDF
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    # Save as PNG
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=600, bbox_inches='tight')
    print(f"Plot saved to {png_file}")
    
    # Also create a combined plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    for idx, (estimator_name, estimates) in enumerate(sorted(all_estimates.items())):
        ax2.scatter(all_true_mi, estimates, alpha=0.6, s=80, label=estimator_name, color=colors[idx], marker='o')
        # ax2.plot(all_true_mi, estimates, linewidth=2, label=estimator_name, color=colors[idx], marker='o', alpha=0.6)

    # Perfect estimation line
    min_val = min(np.min(all_true_mi), np.min([np.min(est) for est in all_estimates.values()]))
    max_val = max(np.max(all_true_mi), np.max([np.max(est) for est in all_estimates.values()]))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2, label='Perfect estimation')

    ax2.set_xlabel('Ground Truth MI')
    ax2.set_ylabel('Estimated MI')
    ax2.set_title('Ground Truth vs Estimated Mutual Information')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    combined_output = output_file.replace('.pdf', '_combined.pdf')
    plt.tight_layout()
    # Save as PDF
    plt.savefig(combined_output, dpi=600, bbox_inches='tight')
    print(f"Combined plot saved to {combined_output}")
    # Save as PNG
    combined_png = combined_output.replace('.pdf', '.png')
    plt.savefig(combined_png, dpi=600, bbox_inches='tight')
    print(f"Combined plot saved to {combined_png}")

    plt.show()  # todo: Delete
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    for estimator_name, estimates in sorted(all_estimates.items()):
        correlation = np.corrcoef(all_true_mi, estimates)[0, 1]
        rmse = np.sqrt(np.mean((all_true_mi - estimates)**2))
        mae = np.mean(np.abs(all_true_mi - estimates))
        print(f"\n{estimator_name}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

if __name__ == "__main__":
    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("Error: No results found!")
    else:
        print(f"Found results for {len(results)} datasets")
        plot_results(results, args.output_file, args.estimators, args.use_final)

