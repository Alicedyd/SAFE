import os
import sys
import yaml
import csv
import torch
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from scipy.special import softmax
from sklearn.metrics import accuracy_score, average_precision_score

from models.resnet import resnet50
import utils
from data.datasets import Get_Transforms


class CustomDataset(Dataset):
    """Dataset for loading real and fake images from paths specified in YAML config"""
    def __init__(self, real_path, fake_path, transform):
        self.transform = transform
        self.data_list = []
        
        # Add real images
        if real_path:
            for img_path in self._get_image_paths(real_path):
                self.data_list.append({"image_path": img_path, "label": 0})
        
        # Add fake images
        if fake_path:
            for img_path in self._get_image_paths(fake_path):
                self.data_list.append({"image_path": img_path, "label": 1})
    
    def _get_image_paths(self, dir_path):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_paths = []
        for root, dirs, files in sorted(os.walk(dir_path)):
            for file in sorted(files):
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, label = sample['image_path'], sample['label']
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, torch.tensor(int(label))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image and the same label in case of error
            return torch.zeros(3, 256, 256), torch.tensor(int(label))


def evaluate(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            all_predictions.append(outputs)
            all_labels.append(labels)
    
    # Concatenate all predictions and labels
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to numpy for metrics calculation
    y_pred = softmax(all_predictions.cpu().numpy(), axis=1)[:, 1]
    y_true = all_labels.cpu().numpy()
    
    # Calculate overall metrics
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    
    # Calculate separate metrics for real and fake images
    real_indices = (y_true == 0)
    fake_indices = (y_true == 1)
    
    # Handle cases where there might be no real or fake images
    real_acc = accuracy_score(y_true[real_indices], (y_pred > 0.5)[real_indices]) if np.any(real_indices) else None
    fake_acc = accuracy_score(y_true[fake_indices], (y_pred > 0.5)[fake_indices]) if np.any(fake_indices) else None
    
    return acc, ap, real_acc, fake_acc


def get_args_parser():
    parser = argparse.ArgumentParser('SAFE Validation', add_help=False)
    
    # Required parameters
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to save results')
    
    # Optional parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--transform_mode', type=str, default='crop', help='Transform mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--dataset_filter', type=str, default=None, help='Only validate on this dataset')
    parser.add_argument('--subset_filter', type=str, default=None, help='Only validate on this subset')
    
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load YAML config
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model and load checkpoint
    model = resnet50(num_classes=2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Get transforms
    class TransformArgs:
        def __init__(self, input_size, transform_mode):
            self.input_size = input_size
            self.transform_mode = transform_mode
            self.jpeg_factor = None
            self.blur_sigma = None
            self.mask_ratio = None
            self.mask_patch_size = None
    
    transform_args = TransformArgs(args.input_size, args.transform_mode)
    _, transform = Get_Transforms(transform_args)
    
    # Results storage - include all metrics
    all_results = []
    all_results.append(["Dataset", "Subset", "Real Images", "Fake Images", "Accuracy (%)", "Average Precision (%)", "Real Accuracy (%)", "Fake Accuracy (%)"])
    
    # Results dictionary for transposed CSV
    result_dict = {
        "Accuracy": [],
        "AP": [],
        "Real Accuracy": [],
        "Fake Accuracy": []
    }
    dataset_names = []
    
    # Process each dataset in the config
    datasets_to_process = [args.dataset_filter] if args.dataset_filter else config.keys()
    
    for dataset_name in datasets_to_process:
        if dataset_name not in config:
            print(f"Warning: Dataset {dataset_name} not found in config. Skipping.")
            continue
        
        dataset_config = config[dataset_name]
        
        # Handle special case for GenEval which only has fake paths
        if dataset_name == "GenEval":
            real_path = None
            subsets_to_process = [args.subset_filter] if args.subset_filter else dataset_config.keys()
            
            for subset_name in subsets_to_process:
                if subset_name not in dataset_config:
                    print(f"Warning: Subset {subset_name} not found in {dataset_name}. Skipping.")
                    continue
                
                fake_path = dataset_config[subset_name].get('fake')
                
                # Skip if no fake path
                if not fake_path:
                    print(f"Warning: No fake path for {dataset_name}/{subset_name}. Skipping.")
                    continue
                
                # We'll create a special case for GenEval where all fake images will be compared against
                # an accuracy target of 100% (since we don't have real images to compare against)
                print(f"Testing {dataset_name}/{subset_name} with fake images only")
                
                # Create dataset with only fake images
                dataset = CustomDataset(None, fake_path, transform)
                
                # Skip if dataset is empty
                if len(dataset) == 0:
                    print(f"Warning: No images found for {dataset_name}/{subset_name}. Skipping.")
                    continue
                
                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False
                )
                
                # Evaluate
                # For GenEval, all images are fake, so overall accuracy equals fake accuracy
                acc, ap, _, fake_acc = evaluate(model, dataloader, device)
                
                # For GenEval, we measure how many images are correctly classified as fake
                # Since there are only fake images, the fake_acc is the key metric
                print(f"{dataset_name}/{subset_name} - Fake Detection Rate: {fake_acc * 100:.2f}%, AP: {ap * 100:.2f}%")
                
                # Store results - for GenEval both accuracy and fake accuracy should be the same
                all_results.append([
                    dataset_name, 
                    subset_name, 
                    0,  # No real images
                    len(dataset),
                    acc * 100,
                    ap * 100,
                    None,  # No real images to calculate real_acc
                    fake_acc * 100
                ])
                
                # Store for transposed CSV
                dataset_key = f"{dataset_name}-{subset_name}"
                dataset_names.append(dataset_key)
                result_dict["Accuracy"].append(acc * 100)
                result_dict["AP"].append(ap * 100)
                result_dict["Real Accuracy"].append(None)
                result_dict["Fake Accuracy"].append(fake_acc * 100)
            
            continue
        
        # Process each subset
        subsets_to_process = [args.subset_filter] if args.subset_filter else [k for k in dataset_config.keys() if k != 'real']
        
        for subset_name in subsets_to_process:
            if subset_name == 'real' or subset_name not in dataset_config:
                continue
            
            # Check if real path is at the top level or nested within the subset
            if dataset_name == "GenImage" and isinstance(dataset_config[subset_name], dict) and 'real' in dataset_config[subset_name]:
                # For GenImage, real path is nested in each subset
                real_path = dataset_config[subset_name].get('real')
            else:
                # For other datasets, real path is at the top level
                real_path = dataset_config.get('real')
            
            # Get fake path for this subset
            if isinstance(dataset_config[subset_name], dict) and 'fake' in dataset_config[subset_name]:
                fake_path = dataset_config[subset_name]['fake']
            else:
                fake_path = None
            
            # Skip if no fake path
            if not fake_path:
                print(f"Warning: No fake path for {dataset_name}/{subset_name}. Skipping.")
                continue
            
            print(f"Testing {dataset_name}/{subset_name}")
            
            # Create dataset
            dataset = CustomDataset(real_path, fake_path, transform)
            
            # Skip if dataset is empty
            if len(dataset) == 0:
                print(f"Warning: No images found for {dataset_name}/{subset_name}. Skipping.")
                continue
            
            # Calculate real and fake counts
            real_count = sum(1 for item in dataset.data_list if item['label'] == 0)
            fake_count = sum(1 for item in dataset.data_list if item['label'] == 1)
            
            # Check if there are any real images
            has_real_images = real_count > 0
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            
            # Evaluate
            acc, ap, real_acc, fake_acc = evaluate(model, dataloader, device)
            
            # Ensure we have real accuracy for GenImage if real images exist
            if dataset_name == "GenImage" and real_count > 0 and real_acc is None:
                print(f"Warning: No real accuracy calculated for {dataset_name}/{subset_name} despite having {real_count} real images")
            
            real_acc_str = f", Real Acc: {real_acc * 100:.2f}%" if real_acc is not None else ""
            fake_acc_str = f", Fake Acc: {fake_acc * 100:.2f}%" if fake_acc is not None else ""
            print(f"{dataset_name}/{subset_name} - Accuracy: {acc * 100:.2f}%, AP: {ap * 100:.2f}%{real_acc_str}{fake_acc_str}")
            
            # Store results
            all_results.append([
                dataset_name, 
                subset_name, 
                real_count,
                fake_count,
                acc * 100,
                ap * 100,
                real_acc * 100 if real_acc is not None else None,
                fake_acc * 100 if fake_acc is not None else None
            ])
            
            # Store for transposed CSV
            dataset_key = f"{dataset_name}-{subset_name}"
            dataset_names.append(dataset_key)
            result_dict["Accuracy"].append(acc * 100)
            result_dict["AP"].append(ap * 100)
            result_dict["Real Accuracy"].append(real_acc * 100 if real_acc is not None else None)
            result_dict["Fake Accuracy"].append(fake_acc * 100 if fake_acc is not None else None)
    
    # Calculate averages for each dataset correctly
    dataset_metrics = defaultdict(lambda: {'acc': [], 'ap': [], 'real_acc': [], 'fake_acc': []})
    for result in all_results[1:]:  # Skip header
        dataset_metrics[result[0]]['acc'].append(result[4])
        dataset_metrics[result[0]]['ap'].append(result[5])
        if result[6] is not None:
            dataset_metrics[result[0]]['real_acc'].append(result[6])
        if result[7] is not None:
            dataset_metrics[result[0]]['fake_acc'].append(result[7])
    
    for dataset, metrics in dataset_metrics.items():
        avg_acc = sum(metrics['acc']) / len(metrics['acc'])
        avg_ap = sum(metrics['ap']) / len(metrics['ap'])
        avg_real_acc = sum(metrics['real_acc']) / len(metrics['real_acc']) if metrics['real_acc'] else None
        avg_fake_acc = sum(metrics['fake_acc']) / len(metrics['fake_acc']) if metrics['fake_acc'] else None
        all_results.append([f"{dataset} Average", "", "", "", avg_acc, avg_ap, avg_real_acc, avg_fake_acc])
    
    # Calculate overall averages - filtering out None values
    all_acc = [result[4] for result in all_results[1:] if isinstance(result[4], (int, float))]
    all_ap = [result[5] for result in all_results[1:] if isinstance(result[5], (int, float))]
    all_real_acc = [result[6] for result in all_results[1:] if isinstance(result[6], (int, float))]
    all_fake_acc = [result[7] for result in all_results[1:] if isinstance(result[7], (int, float))]
    
    overall_acc = sum(all_acc) / len(all_acc)
    overall_ap = sum(all_ap) / len(all_ap)
    overall_real_acc = sum(all_real_acc) / len(all_real_acc) if all_real_acc else None
    overall_fake_acc = sum(all_fake_acc) / len(all_fake_acc) if all_fake_acc else None
    
    all_results.append(["Overall Average", "", "", "", overall_acc, overall_ap, overall_real_acc, overall_fake_acc])
    
    # Save standard results to CSV
    checkpoint_name = os.path.basename(args.checkpoint).split('.')[0]
    csv_path = os.path.join(args.output_dir, f"{checkpoint_name}_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_results)
    
    # Save transposed results to CSV
    transposed_csv_path = os.path.join(args.output_dir, f"{checkpoint_name}_results_transposed.csv")
    with open(transposed_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Metric"] + dataset_names)
        # Write each metric row
        for metric in ["Accuracy", "Real Accuracy", "Fake Accuracy"]:  # Removed AP as requested
            if metric in result_dict:
                writer.writerow([metric] + result_dict[metric])
    
    print(f"Standard results saved to {csv_path}")
    print(f"Transposed results saved to {transposed_csv_path}")


if __name__ == "__main__":
    main()