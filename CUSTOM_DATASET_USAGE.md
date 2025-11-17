# Custom Dataset Benchmark Usage Guide

This guide explains how to use the modified codebase to benchmark MI estimators on your custom dataset.

## Overview

The custom dataset benchmark consists of three main components:

1. **`libs/utils_custom.py`**: Utility functions to load your custom npz dataset
2. **`benchmark_custom.py`**: Script to run all MI estimators on your dataset
3. **`plot_results.py`**: Script to visualize true MI vs estimated MI

## Dataset Format

Your dataset should be organized as follows:
```
dataset/datasets_4624753/
├── folder1/
│   └── dataset_file.npz
├── folder2/
│   └── dataset_file.npz
└── ...
```

Each `.npz` file should contain:
- **`X`**: Array of shape `(10000, 2, 3, 32, 32)` where:
  - First dimension: 10,000 paired samples
  - Second dimension: 2 (X and Y paired images)
  - Remaining dimensions: 3 channels, 32x32 pixels (CIFAR-like images)
- **`MI_results`**: Array where the 3rd value (index 2) is the known true MI value

## Running the Benchmark

### Step 1: Run all estimators on your dataset

```bash
python benchmark_custom.py \
    --gpu_id 0 \
    --dataset_dir dataset/datasets_4624753 \
    --savepath results/custom_benchmark \
    --estimator all \
    --batch_size 64 \
    --n_steps 20000 \
    --learning_rate 0.0005
```

**Key Arguments:**
- `--gpu_id`: GPU ID to use (set to -1 for CPU)
- `--dataset_dir`: Path to your dataset directory
- `--savepath`: Where to save results
- `--estimator`: `all` to run all estimators, or specify one (e.g., `dv`, `mine`, `smile-5`)
- `--batch_size`: Batch size for training
- `--n_steps`: Number of training steps
- `--output_scale`: `bit` or `nat` (default: `bit`)

**Available Estimators:**
- `nwj`: Nguyen-Wainwright-Jordan
- `js`: Jensen-Shannon
- `infonce`: InfoNCE
- `dv`: Donsker-Varadhan
- `mine`: MINE
- `smile-1`: SMILE with clip=1
- `smile-5`: SMILE with clip=5
- `smile-inf`: SMILE with clip=infinity

**Critic Architecture Options:**
- `--critic_type`: `joint`, `separable`, `bilinear`, or `inner` (default: `joint`)
- `--critic_depth`: Depth of MLP critic (default: 2)
- `--critic_width`: Width of MLP critic (default: 256)
- `--critic_embed`: Embedding size for separable critic (default: 32)

### Step 2: Plot results

After running the benchmark, generate plots:

```bash
python plot_results.py \
    --results_dir results/custom_benchmark \
    --output_file results/custom_benchmark/mi_comparison.pdf
```

**Additional Options:**
- `--estimators`: Specify which estimators to plot (e.g., `--estimators dv mine smile-5`)
- `--use_final`: Use average of last 1000 steps instead of full average

## Output

The benchmark script will create:
- Individual result files: `results/custom_benchmark/{dataset_folder}/{estimator}/mi.npy`
- Summary file: `results/custom_benchmark/summary.npz` (contains all statistics)
- Log file: `results/custom_benchmark/logs.log`

The plotting script will create:
- Individual plots: `results/custom_benchmark/mi_comparison.pdf` (one subplot per estimator)
- Combined plot: `results/custom_benchmark/mi_comparison_combined.pdf` (all estimators together)

## Example Workflow

1. **Run benchmark on all datasets with all estimators:**
   ```bash
   python benchmark_custom.py --gpu_id 0 --estimator all
   ```

2. **Plot results:**
   ```bash
   python plot_results.py
   ```

3. **Run only specific estimators:**
   ```bash
   python benchmark_custom.py --estimator dv --estimator mine
   ```

4. **Use different critic architecture:**
   ```bash
   python benchmark_custom.py --critic_type separable --critic_width 512
   ```

## Notes

- The true MI values are automatically extracted from the `MI_results` array (3rd value)
- Results are saved in both bit and nat scales (controlled by `--output_scale`)
- The script processes all subdirectories in the dataset directory automatically
- Each estimator is trained independently for each dataset

## Troubleshooting

**Issue**: "No datasets found!"
- Check that your dataset directory path is correct
- Ensure each subdirectory contains at least one `.npz` file

**Issue**: "Could not extract true MI"
- Verify that your `.npz` files contain `MI_results` array
- Check that the 3rd value (index 2) exists in `MI_results`

**Issue**: Out of memory
- Reduce `--batch_size`
- Reduce `--n_steps`
- Use CPU mode with `--gpu_id -1`

