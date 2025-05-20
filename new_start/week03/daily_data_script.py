import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Tuple, Optional

# Constants
# DAILY_DIR_PATH = "/Users/kuba/Documents/data/raw/kuba_twp5_watchdata_nov19" 
DAILY_DIR_PATH = "/home/kuba/Documents/data/raw/kuba_watch_data"
DAILY_CLASS_LABEL = 0

class DailyDataset(Dataset):
    """Dataset for daily living activity sensor data."""
    
    def __init__(self, X, y, flatten=False):
        """
        Initialize the dataset.
        
        Args:
            X: Sensor data tensor. 
               If flatten=False: shape (n_windows, 6, window_size)
               If flatten=True: shape (n_windows, 2, 3*window_size)
            y: Labels tensor with shape (n_windows, 1)
            flatten: Whether data is flattened
        """
        self.X = X
        self.y = y
        self.flatten = flatten

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return the entire X tensor for this index and the corresponding label
        return self.X[idx], self.y[idx]


def window_maker(data, window_size, stride, flatten):
    """Create windows from time series data."""
    windows = []
    
    if flatten:
        # Flatten x, y, z channels into a single vector per window
        for i in range(0, len(data['x']) - window_size + 1, stride):
            combined = []
            combined.extend(data['x'][i:i + window_size].tolist())
            combined.extend(data['y'][i:i + window_size].tolist())
            combined.extend(data['z'][i:i + window_size].tolist())
            windows.append(combined)
    else:
        # Keep x, y, z as separate channels
        for i in range(0, len(data['x']) - window_size + 1, stride):
            window = [
                data['x'][i:i + window_size].tolist(),
                data['y'][i:i + window_size].tolist(),
                data['z'][i:i + window_size].tolist()
            ]
            windows.append(window)
            
    return windows


def calculate_magnitude(acc, gyro, flatten):
    """Calculate magnitude of acceleration and gyroscope data."""
    if not flatten:
        # Shape: (num_windows, 3, window_size)
        acc_magnitudes = torch.sqrt(acc[:, 0]**2 + acc[:, 1]**2 + acc[:, 2]**2)
        gyro_magnitudes = torch.sqrt(gyro[:, 0]**2 + gyro[:, 1]**2 + gyro[:, 2]**2)
    else:
        # Shape: (num_windows, 3*window_size)
        acc_magnitudes = torch.sqrt(acc**2)
        gyro_magnitudes = torch.sqrt(gyro**2)
        
    return acc_magnitudes, gyro_magnitudes


def filter_static_windows(acc, gyro, flatten, acc_threshold=0.05, gyro_threshold=0.02):
    """Remove windows with little or no movement."""
    acc_mag, gyro_mag = calculate_magnitude(acc, gyro, flatten)

    # Calculate standard deviation - low std dev means little variation
    acc_std = torch.std(acc_mag, dim=1)
    gyro_std = torch.std(gyro_mag, dim=1)
    
    # Keep windows with enough movement
    valid_indices = torch.logical_or(acc_std > acc_threshold, gyro_std > gyro_threshold)
    
    filtered_acc = acc[valid_indices]
    filtered_gyro = gyro[valid_indices]
    
    if len(filtered_acc) == 0 or len(filtered_gyro) == 0:
        print("Warning: All windows were filtered out due to lack of movement")
        
    return filtered_acc, filtered_gyro


def process_daily_file(file_name, window_size, stride, flatten, filter_static=True):
    """Extract windows from a daily activity recording file."""
    file_path = os.path.join(DAILY_DIR_PATH, file_name)
    
    # Load acceleration data
    acc_path = os.path.join(file_path, "acceleration.csv")
    acc_data = pd.read_csv(acc_path, skiprows=1)
    acc_data['timestamp'] = (acc_data['timestamp'] - acc_data['timestamp'].iloc[0]) * 1e-9
    
    # Load gyroscope data
    gyro_path = os.path.join(file_path, "gyroscope.csv")
    gyro_data = pd.read_csv(gyro_path, skiprows=1)
    gyro_data['timestamp'] = (gyro_data['timestamp'] - gyro_data['timestamp'].iloc[0]) * 1e-9
    
    # Create windows
    acc_windows = window_maker(acc_data, window_size, stride, flatten)
    gyro_windows = window_maker(gyro_data, window_size, stride, flatten)
    
    # Convert to tensors
    acc_tensor = torch.tensor(acc_windows)
    gyro_tensor = torch.tensor(gyro_windows)

    # sometimes we can end up with one more or less samples becasue the watch does not perfectly sample data at the same time
    # this way we make sure that they have the same number of samples
    acc_tensor = acc_tensor[:min(len(acc_tensor), len(gyro_tensor))]
    gyro_tensor = gyro_tensor[:min(len(acc_tensor), len(gyro_tensor))]

    if len(acc_tensor) == 0: # We just need to check if acc == 0 becasue abocve we set their lens to the min of the two
        return torch.tensor([]), torch.tensor([])
    
    # Filter out static windows if requested
    if filter_static:
        acc_tensor, gyro_tensor = filter_static_windows(acc_tensor, gyro_tensor, flatten)
    
    # Combine data according to format
    if not flatten:
        # Combine acc and gyro to shape (n_windows, 6, window_size)
        X = torch.cat((acc_tensor, gyro_tensor), dim=1)
    else:
        # Stack acc and gyro to shape (n_windows, 2*3*window_size)
        X = torch.cat((acc_tensor, gyro_tensor), dim=1)
        # X = torch.stack((acc_tensor, gyro_tensor), dim=1)
    
    # Create labels (all daily activities have the same label)
    y = torch.tensor([[DAILY_CLASS_LABEL]] * len(X))
    
    return X, y


def collect_windows(num_windows_needed, available_dirs, window_size, stride, flatten, filter_static=True):
    """Collect specified number of windows from available files."""
    collected_X = []
    remaining_files = available_dirs.copy()
    
    if not remaining_files:
        raise ValueError("No files available to extract windows from")
    
    windows_collected = 0
    
    while windows_collected < num_windows_needed and remaining_files:
        # Take the next file
        file_name = remaining_files.pop(0)
        
        # Process the file
        try:
            X, _ = process_daily_file(file_name, window_size, stride, flatten, filter_static)

            if len(X) == 0: # We were not able to sample any windows from the file
                continue
            
            # If this file provides enough windows to meet the requirement
            if windows_collected + len(X) >= num_windows_needed:
                # Take only as many as needed
                needed = num_windows_needed - windows_collected
                collected_X.append(X[:needed])
                windows_collected += needed
                break
            else:
                # Take all windows from this file
                collected_X.append(X)
                windows_collected += len(X)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue
    
    if windows_collected < num_windows_needed:
        print(f"Warning: Could only collect {windows_collected} windows, fewer than the {num_windows_needed} requested")
    
    # Combine all collected windows
    if collected_X:
        X_combined = torch.cat(collected_X, dim=0)
        y_combined = torch.tensor([[DAILY_CLASS_LABEL]] * len(X_combined))
        return X_combined, y_combined, remaining_files
    else:
        raise ValueError("Failed to collect any windows")


def load_daily_data(num_train_samples, num_dev_samples, num_test_samples, 
                   window_size=400, stride=10, flatten=False, filter_static=True):
    """
    Load daily activity data with the specified number of samples for each split.
    
    Args:
        num_train_samples: Number of windows for training set
        num_dev_samples: Number of windows for dev/validation set
        num_test_samples: Number of windows for test set
        window_size: Size of the window in samples
        stride: Stride between consecutive windows
        flatten: Whether to flatten x,y,z channels
        filter_static: Whether to filter out windows with little movement
        
    Returns:
        Dictionary with train, dev, and test datasets
    """
    # Get all available dirs
    available_dirs = [d for d in sorted(os.listdir(DAILY_DIR_PATH)) if os.path.isdir(os.path.join(DAILY_DIR_PATH, d))]    
    
    # Set random seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(available_dirs)
    
    print('All daily dir:',available_dirs)
    # Collect windows for each split
    try:
        # Training set
        train_X, train_y, available_dirs = collect_windows(
            num_train_samples, available_dirs, window_size, stride, flatten, filter_static
        )
        train_dataset = DailyDataset(train_X, train_y, flatten)
        print('Daily dir after train:',available_dirs)
        
        
        # Dev set
        dev_X, dev_y, available_dirs = collect_windows(
            num_dev_samples, available_dirs, window_size, stride, flatten, filter_static
        )
        dev_dataset = DailyDataset(dev_X, dev_y, flatten)
        print('Daily dir after dev:',available_dirs)

        
        # Test set
        test_X, test_y, available_dirs = collect_windows(
            num_test_samples, available_dirs, window_size, stride, flatten, filter_static
        )
        test_dataset = DailyDataset(test_X, test_y, flatten)
        print('Daily dir for test:',available_dirs)

        
        return {
            'train': train_dataset,
            'dev': dev_dataset,
            'test': test_dataset,
            'remaining_files': available_dirs
        }
    
    except Exception as e:
        print(f"Error loading daily data: {e}")
        return {
            'train': None,
            'dev': None,
            'test': None,
            'error': str(e)
        }


if __name__ == "__main__":
    # Example usage
    data = load_daily_data(
        num_train_samples=100,
        num_dev_samples=20,
        num_test_samples=30,
        window_size=400,
        stride=200,
        flatten=False
    )
    
    print(f"Training daily set size: {len(data['train'])}")
    print(f"Dev daily set size: {len(data['dev'])}")
    print(f"Test daily set size: {len(data['test'])}")
    
    # Demonstrate data shapes
    sample_X, sample_y = data['train'][0]
    print(f"Sample X shape: {sample_X.shape}")
    print(f"Sample y shape: {sample_y.shape}")
    print(f"Sample y value: {sample_y}")
    
    # Report remaining files
    print(f"Remaining files: {len(data['remaining_files'])}")