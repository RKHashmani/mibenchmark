import torch
import numpy as np
import os
import glob

class CustomDataset:
    """Custom dataset loader for npz files with paired images and known MI values."""
    
    def __init__(self, dataset_dir):
        """
        Initialize custom dataset loader.
        
        Args:
            dataset_dir: Path to directory containing subdirectories with npz files
        """
        self.dataset_dir = dataset_dir
        self.datasets = {}
        self.true_mi_values = {}
        
        # Find all npz files in subdirectories
        subdirs = sorted([d for d in os.listdir(dataset_dir) 
                         if os.path.isdir(os.path.join(dataset_dir, d))])
        
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_dir, subdir)
            npz_files = glob.glob(os.path.join(subdir_path, "*.npz"))
            
            if npz_files:
                # Load the first npz file found in each subdirectory
                npz_path = npz_files[0]
                data = np.load(npz_path)
                
                # Extract X and MI_results
                X = data['X']  # Shape: (10000, 2, 3, 32, 32)
                MI_results = data['MI_results']  # Get the 3rd value (index 2)
                true_mi = MI_results[2] if len(MI_results) > 2 else MI_results[0]
                
                # Store dataset
                self.datasets[subdir] = {
                    'X': X,
                    'true_mi': true_mi,
                    'n_samples': X.shape[0]
                }
                self.true_mi_values[subdir] = true_mi
                
                print(f"Loaded {subdir}: {X.shape[0]} samples, True MI = {true_mi:.4f}")
    
    def get_dataset(self, subdir_name):
        """Get dataset for a specific subdirectory."""
        return self.datasets[subdir_name]
    
    def get_all_datasets(self):
        """Get all loaded datasets."""
        return self.datasets
    
    def get_all_true_mi(self):
        """Get all true MI values."""
        return self.true_mi_values


def custom_image_batch(dataset_dict, batch_size=64, seed=None):
    """
    Generate a batch of paired images from custom dataset.
    
    Args:
        dataset_dict: Dictionary with 'X' key containing (n_samples, 2, 3, 32, 32) array
        batch_size: Number of samples in batch
        seed: Random seed for reproducibility
    
    Returns:
        z1, z2: Two tensors of shape (batch_size, 3, 32, 32)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = dataset_dict['X']  # Shape: (n_samples, 2, 3, 32, 32)
    n_samples = X.shape[0]
    
    # Randomly sample indices
    indices = np.random.choice(n_samples, batch_size, replace=True)
    
    # Extract X and Y pairs
    z1 = torch.from_numpy(X[indices, 0, :, :, :]).float()  # X images
    z2 = torch.from_numpy(X[indices, 1, :, :, :]).float()  # Y images
    
    return z1, z2

