import os
import argparse
import logging
import numpy as np
import torch
from libs.bounds import estimate_mutual_information
from libs.critics import set_critic
from libs.utils_custom import CustomDataset, custom_image_batch
from libs.encoder import irevnet, realnvp, maf, pretrained_resnet
from utils import nat2bit, bit2nat

# Initialize argument parser
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument("--gpu_id", type=int, default=0)  # For CPU implementation, set -1
parser.add_argument("--dataset_dir", type=str, default="dataset/datasets_4624753", help="Directory containing custom datasets")
parser.add_argument("--savepath", type=str, default="results/custom_benchmark")

parser.add_argument("--dr", type=int, default=3072, help="representation dimension (3*32*32=3072 for CIFAR-like images)")

parser.add_argument("--output_scale", type=str, default="bit", choices=["bit", "nat"])
parser.add_argument("--input_scale", type=str, default="bit", choices=["bit", "nat"], 
                    help="Scale of true MI values in your dataset (bit or nat)")

parser.add_argument("--critic_type", type=str, default="joint", choices=["joint", "separable", "bilinear", "inner"])
parser.add_argument("--critic_depth", type=int, default=2)
parser.add_argument("--critic_width", type=int, default=256)
parser.add_argument("--critic_embed", type=int, default=32)  # Only for separable critic
parser.add_argument("--estimator", type=str, default="all", 
                    choices=["all", "nwj", "js", "infonce", "dv", "mine", "smile-1", "smile-5", "smile-inf"])

parser.add_argument("--encoder", type=str, default="None", choices=["None", "irevnet", "realnvp", "maf", "pretrained_resnet"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.0005)
parser.add_argument("--n_steps", type=int, default=20000)

args = parser.parse_args()

# List of all estimators
ALL_ESTIMATORS = ["nwj", "js", "infonce", "dv", "mine", "smile-1", "smile-5", "smile-inf"]

def main():
    # Set up logging
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
        
    file_handler = logging.FileHandler(os.path.join(args.savepath, "logs.log"))
    log.addHandler(file_handler)
    log.info(f'Logs will be saved at.. {args.savepath}')
    
    # Set GPU or CPU environment
    if args.gpu_id >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    device = 'cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    log.info(f'Using device: {device}')
    
    # Load custom datasets
    log.info(f'Loading datasets from {args.dataset_dir}')
    custom_dataset = CustomDataset(args.dataset_dir)
    all_datasets = custom_dataset.get_all_datasets()
    all_true_mi = custom_dataset.get_all_true_mi()
    
    if len(all_datasets) == 0:
        log.error("No datasets found!")
        return
    
    log.info(f'Found {len(all_datasets)} datasets')
    
    # Determine which estimators to run
    if args.estimator == "all":
        estimators_to_run = ALL_ESTIMATORS
    else:
        estimators_to_run = [args.estimator]
    
    # Initialize encoder if specified
    encoder_fn = None
    if args.encoder != "None":
        # Map encoder names to functions
        encoder_map = {
            "irevnet": irevnet,
            "realnvp": realnvp,
            "maf": maf,
            "pretrained_resnet": pretrained_resnet
        }
        
        encoder_func = encoder_map.get(args.encoder)
        if encoder_func is None:
            raise ValueError(f"Unknown encoder: {args.encoder}")
        
        if args.encoder in ["irevnet", "pretrained_resnet"]:
            encoder_fn = encoder_func((3, 32, 32)).to(device)
        else:  # realnvp, maf
            encoder_fn = encoder_func(args.dr).to(device)
    
    # Results storage
    all_results = {}
    
    # Process each dataset
    for subdir_name, dataset_dict in sorted(all_datasets.items()):
        true_mi_raw = dataset_dict['true_mi']  # Raw value from dataset
        
        # Convert true_mi to the scale we'll use for comparison
        if args.input_scale == "nat" and args.output_scale == "bit":
            # Input is nats, output is bits - convert
            true_mi = nat2bit(true_mi_raw)
        elif args.input_scale == "bit" and args.output_scale == "nat":
            # Input is bits, output is nats - convert
            true_mi = bit2nat(true_mi_raw)
        else:
            # Same scale, no conversion needed
            true_mi = true_mi_raw
        
        log.info(f'\n{"="*60}')
        log.info(f'Processing dataset: {subdir_name}')
        log.info(f'True MI (raw from file): {true_mi_raw:.4f} {args.input_scale}')
        log.info(f'True MI (for comparison): {true_mi:.4f} {args.output_scale}')
        log.info(f'{"="*60}')
        
        dataset_results = {}
        dataset_results['true_mi'] = true_mi
        dataset_results['estimates'] = {}
        
        # Run each estimator
        for estimator_name in estimators_to_run:
            log.info(f'\nRunning estimator: {estimator_name}')
            
            # Set critic based on encoder type
            if args.encoder in ["None", "maf", "realnvp"]:
                critic = set_critic(args.critic_type, args.dr, 
                                  hidden_dim=args.critic_width, 
                                  embed_dim=args.critic_embed, 
                                  layers=args.critic_depth, 
                                  device=device)
            elif args.encoder == "irevnet":
                critic = set_critic(args.critic_type, 3072*2*2, 
                                  hidden_dim=args.critic_width, 
                                  embed_dim=args.critic_embed, 
                                  layers=args.critic_depth, 
                                  device=device)
            elif args.encoder == "pretrained_resnet":
                critic = set_critic(args.critic_type, 2048, 
                                  hidden_dim=args.critic_width, 
                                  embed_dim=args.critic_embed, 
                                  layers=args.critic_depth, 
                                  device=device)
            
            # Set optimizer
            opt_crit = torch.optim.Adam(critic.parameters(), lr=args.learning_rate)
            
            # MI parameters
            mi_params = dict(estimator=estimator_name, critic=args.critic_type, baseline='unnormalized')
            
            # Training loop
            estimates = []
            buffer = None
            
            for i in range(args.n_steps):
                torch.manual_seed(i)
                
                # Generate batch
                z1, z2 = custom_image_batch(dataset_dict, batch_size=args.batch_size, seed=i)
                
                # Apply encoder if specified
                with torch.no_grad():
                    if args.encoder == "irevnet":
                        _, z1 = encoder_fn(z1.to(device))
                        _, z2 = encoder_fn(z2.to(device))
                    elif args.encoder in ["realnvp", "maf"]:
                        z1, _ = encoder_fn(z1.view(args.batch_size, -1).to(device))
                        z2, _ = encoder_fn(z2.view(args.batch_size, -1).to(device))
                    elif args.encoder == "pretrained_resnet":
                        z1 = encoder_fn(z1.to(device))
                        z2 = encoder_fn(z2.to(device))
                
                # Training step
                opt_crit.zero_grad()
                
                # Flatten images for critic
                z1_flat = z1.view(args.batch_size, -1).to(device)
                z2_flat = z2.view(args.batch_size, -1).to(device)
                
                if mi_params['estimator'] == "mine":
                    mi, buffer = estimate_mutual_information(
                        mi_params['estimator'], 
                        z1_flat, 
                        z2_flat, 
                        critic, 
                        baseline_fn=None, 
                        buffer=buffer
                    )
                else:
                    mi = estimate_mutual_information(
                        mi_params['estimator'], 
                        z1_flat, 
                        z2_flat, 
                        critic, 
                        baseline_fn=None, 
                        buffer=None
                    )
                    buffer = None
                
                loss = -mi
                loss.backward()
                opt_crit.step()
                
                # Convert to output scale
                mi_val = mi.detach().cpu().numpy()
                if args.output_scale == "bit":
                    mi_val = nat2bit(mi_val)
                # true_mi is already in the correct scale from the conversion above
                true_mi_display = true_mi
                
                estimates.append(mi_val)
                
                if i % 1000 == 0:
                    avg_est = np.mean(estimates)
                    log.info(f'STEP: {i}, Truth: {true_mi_display:.4f}, Estimated: {mi_val:.4f}, Average: {avg_est:.4f}')
            
            # Store results
            estimates_array = np.array(estimates)
            dataset_results['estimates'][estimator_name] = estimates_array
            
            # Save individual result
            estimator_savepath = os.path.join(args.savepath, subdir_name, estimator_name)
            os.makedirs(estimator_savepath, exist_ok=True)
            np.save(os.path.join(estimator_savepath, "mi.npy"), estimates_array)
            
            log.info(f'Final average estimate for {estimator_name}: {np.mean(estimates_array):.4f}')
        
        all_results[subdir_name] = dataset_results
    
    # Save summary results
    summary_path = os.path.join(args.savepath, "summary.npz")
    summary_data = {}
    dataset_names = list(all_results.keys())
    summary_data['dataset_names'] = np.array(dataset_names, dtype=object)
    summary_data['true_mi_values'] = np.array([all_results[k]['true_mi'] for k in dataset_names])
    
    for subdir_name, results in all_results.items():
        summary_data[f'{subdir_name}_true_mi'] = results['true_mi']
        for estimator_name, estimates in results['estimates'].items():
            summary_data[f'{subdir_name}_{estimator_name}_mean'] = np.mean(estimates)
            summary_data[f'{subdir_name}_{estimator_name}_std'] = np.std(estimates)
            summary_data[f'{subdir_name}_{estimator_name}_final'] = np.mean(estimates[-1000:])  # Last 1000 steps
    
    np.savez(summary_path, **summary_data)
    log.info(f'\nSummary saved to {summary_path}')
    log.info("======================================== END BENCHMARK ========================================")
    
    return all_results

if __name__ == "__main__":
    main()

