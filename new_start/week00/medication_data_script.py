import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from enum import Enum
from typing import List, Dict, Optional, Union

# Constants
RAW_DATA_DIR = '/Users/kuba/Documents/data/raw/medicationTakingData/processed/3_final'
MEDICINE_CLASS_LABEL = 0

class LeakageOption(Enum):
    """Controls how data is split between train/test sets."""
    NO_LEAKAGE = 0     # No participant appears in both train and test
    PARTIAL_LEAKAGE = 1  # Same participant but different recordings may appear in train/test
    FULL_LEAKAGE = 2   # Random split of all data

# Label mappings
LABEL_MAPPING = [
    ('left', 'water', 'leftWater'),
    ('left', 'listerine', 'leftLister'),
    ('right', 'water', 'rightWater'),
    ('right', 'listerine', 'rightLister')
]

LABELS_TO_ONE_HOT = {
    'leftWater': [MEDICINE_CLASS_LABEL],
    'leftLister': [MEDICINE_CLASS_LABEL],
    'rightWater': [MEDICINE_CLASS_LABEL],
    'rightLister': [MEDICINE_CLASS_LABEL]
}

class MedicationDataset(Dataset):
    """Dataset for medication-taking sensor data."""
    
    def __init__(self, X, y, flatten=False):
        self.X = X
        self.y = y
        self.flatten = flatten

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def window_maker(data, window_size, stride, flatten):
    """Create windows from time series data."""
    windows = []
    
    if flatten:
        # Flatten x, y, z channels into a single vector per window
        for i in range(0, len(data['x']) - window_size + 1, stride):
            window = []
            window.extend(data['x'][i:i + window_size].tolist())
            window.extend(data['y'][i:i + window_size].tolist())
            window.extend(data['z'][i:i + window_size].tolist())
            windows.append(window)
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


def process_recording(acc_data, gyro_data, labels, window_size, stride, flatten):
    """Extract activity windows and labels from a recording."""
    acc_windows, gyro_windows, labels_list = [], [], []
    
    for side, liquid, label_key in LABEL_MAPPING:
        if side in labels and liquid in labels[side]:
            for label in labels[side][liquid]:
                # Filter data within the labeled time range
                filtered_acc = acc_data[(acc_data.timestamp > label['start']) & 
                                       (acc_data.timestamp < label['end'])]
                filtered_gyro = gyro_data[(gyro_data.timestamp > label['start']) & 
                                         (gyro_data.timestamp < label['end'])]

                # Only process if we have enough data for at least one window
                if len(filtered_acc) >= window_size:
                    # Create windows
                    acc_windows_batch = window_maker(filtered_acc, window_size, stride, flatten)
                    gyro_windows_batch = window_maker(filtered_gyro, window_size, stride, flatten)

                    # Handle potential mismatch in window counts
                    min_windows = min(len(acc_windows_batch), len(gyro_windows_batch))
                    acc_windows_batch = acc_windows_batch[:min_windows]
                    gyro_windows_batch = gyro_windows_batch[:min_windows]

                    # Add windows and labels
                    acc_windows.extend(acc_windows_batch)
                    gyro_windows.extend(gyro_windows_batch)
                    labels_list.extend([LABELS_TO_ONE_HOT[label_key]] * min_windows)
    
    return acc_windows, gyro_windows, labels_list


def process_participant(participant_dir, window_size, stride, flatten):
    """Process all recordings for a participant."""
    acc_windows_all, gyro_windows_all, labels_all = [], [], []

    # Process each recording
    for recording in sorted(os.listdir(participant_dir)):
        if recording == '.DS_Store':
            continue
            
        recording_path = os.path.join(participant_dir, recording)
        
        # Load labels
        with open(os.path.join(recording_path, 'labels.json')) as f:
            labels = json.load(f)
            
        # Load sensor data
        acc_data = pd.read_csv(os.path.join(recording_path, 'acceleration.csv'), skiprows=1)
        acc_data['timestamp'] = (acc_data['timestamp'] - acc_data['timestamp'].iloc[0]) * 1e-9
        
        gyro_data = pd.read_csv(os.path.join(recording_path, 'gyroscope.csv'), skiprows=1)
        gyro_data['timestamp'] = (gyro_data['timestamp'] - gyro_data['timestamp'].iloc[0]) * 1e-9
        
        # Process the recording
        acc_windows, gyro_windows, window_labels = process_recording(
            acc_data, gyro_data, labels, window_size, stride, flatten
        )
        
        # Only add if we got windows
        if acc_windows:
            acc_windows_all.extend(acc_windows)
            gyro_windows_all.extend(gyro_windows)
            labels_all.extend(window_labels)
    
    # Convert to tensors and combine according to the specified format
    if not flatten:
        # Combine acc and gyro to shape (n_windows, 6, window_size)
        acc_tensor = torch.tensor(acc_windows_all)  # shape: (n_windows, 3, window_size)
        gyro_tensor = torch.tensor(gyro_windows_all)  # shape: (n_windows, 3, window_size)
        X = torch.cat((acc_tensor, gyro_tensor), dim=1)  # shape: (n_windows, 6, window_size)
    else:
        # Stack acc and gyro to shape (n_windows, 2, 3*window_size)
        acc_tensor = torch.tensor(acc_windows_all)  # shape: (n_windows, 3*window_size)
        gyro_tensor = torch.tensor(gyro_windows_all)  # shape: (n_windows, 3*window_size)
        X = torch.stack((acc_tensor, gyro_tensor), dim=1)  # shape: (n_windows, 2, 3*window_size)
    
    return X, torch.tensor(labels_all)


def get_directory_splits(leakage_option, train_p=0.8, dev_p=0.1, test_p=0.1,
                         specific_participants=None):
    """Create train/dev/test splits based on leakage settings."""
    # Validate proportions
    if abs(train_p + dev_p + test_p - 1.0) > 1e-10:
        raise ValueError("Split proportions must sum to 1.0")

    # Get participant directories
    all_dirs = [d for d in sorted(os.listdir(RAW_DATA_DIR)) if d != '.DS_Store']
    
    # Filter for specific participants if requested
    if specific_participants:
        all_dirs = [d for d in all_dirs if d in specific_participants]
        if not all_dirs:
            raise ValueError("No specified participants found in the data directory")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if leakage_option == LeakageOption.NO_LEAKAGE:
        # Split at participant level
        np.random.shuffle(all_dirs)
        n_train = int(len(all_dirs) * train_p)
        n_dev = int(len(all_dirs) * dev_p)
        
        return {
            'train_dirs': all_dirs[:n_train],
            'dev_dirs': all_dirs[n_train:n_train + n_dev],
            'test_dirs': all_dirs[n_train + n_dev:]
        }
    
    elif leakage_option == LeakageOption.PARTIAL_LEAKAGE:
        # Split by recordings
        all_recordings = []
        
        # Collect all recordings
        for participant in all_dirs:
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            recordings = [r for r in os.listdir(participant_path) if r != '.DS_Store']
            all_recordings.extend([(participant, r) for r in recordings])
        
        np.random.shuffle(all_recordings)
        n_train = int(len(all_recordings) * train_p)
        n_dev = int(len(all_recordings) * dev_p)
        
        return {
            'train_recordings': all_recordings[:n_train],
            'dev_recordings': all_recordings[n_train:n_train + n_dev],
            'test_recordings': all_recordings[n_train + n_dev:]
        }
    
    else:  # LeakageOption.FULL_LEAKAGE
        return {'all_dirs': all_dirs}


def load_medication_data(window_size=400, stride=10, flatten=False, train_p=0.8, 
                         dev_p=0.1, test_p=0.1, leakage_option=LeakageOption.NO_LEAKAGE,
                         specific_participants=None):
    """
    Load medication data with configurable options.
    
    Args:
        window_size: Size of the window in samples
        stride: Stride between consecutive windows
        flatten: Whether to flatten x,y,z channels
        train_p: Proportion for training set
        dev_p: Proportion for dev/validation set  
        test_p: Proportion for test set
        leakage_option: How to handle data leakage between splits
        specific_participants: List of specific participant IDs to include
        
    Returns:
        Dictionary with train, dev, and test datasets
    """
    # Get directory splits based on leakage option
    dir_splits = get_directory_splits(
        leakage_option, train_p, dev_p, test_p, specific_participants
    )
    
    if leakage_option == LeakageOption.NO_LEAKAGE:
        # No leakage: separate participants in train/dev/test
        train_datasets, dev_datasets, test_datasets = [], [], []
        
        # Process train participants
        for participant in dir_splits['train_dirs']:
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            X, y = process_participant(participant_path, window_size, stride, flatten)
            if len(y) > 0:
                train_datasets.append(MedicationDataset(X, y, flatten))
        
        # Process dev participants
        for participant in dir_splits['dev_dirs']:
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            X, y = process_participant(participant_path, window_size, stride, flatten)
            if len(y) > 0:
                dev_datasets.append(MedicationDataset(X, y, flatten))
        
        # Process test participants
        for participant in dir_splits['test_dirs']:
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            X, y = process_participant(participant_path, window_size, stride, flatten)
            if len(y) > 0:
                test_datasets.append(MedicationDataset(X, y, flatten))
        
    elif leakage_option == LeakageOption.PARTIAL_LEAKAGE:
        # Partial leakage: split by recordings
        train_datasets, dev_datasets, test_datasets = [], [], []
        
        # Group recordings by participant and split
        participant_recordings = {}
        for participant, recording in dir_splits['train_recordings']:
            if participant not in participant_recordings:
                participant_recordings[participant] = {'train': [], 'dev': [], 'test': []}
            participant_recordings[participant]['train'].append(recording)
            
        for participant, recording in dir_splits['dev_recordings']:
            if participant not in participant_recordings:
                participant_recordings[participant] = {'train': [], 'dev': [], 'test': []}
            participant_recordings[participant]['dev'].append(recording)
            
        for participant, recording in dir_splits['test_recordings']:
            if participant not in participant_recordings:
                participant_recordings[participant] = {'train': [], 'dev': [], 'test': []}
            participant_recordings[participant]['test'].append(recording)
        
        # Process each participant's recordings by split
        for participant, split_recordings in participant_recordings.items():
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            
            for split_name, recordings in [
                ('train', split_recordings['train']),
                ('dev', split_recordings['dev']),
                ('test', split_recordings['test'])
            ]:
                if not recordings:
                    continue
                
                split_acc, split_gyro, split_labels = [], [], []
                
                # Process recordings for this split
                for recording in recordings:
                    recording_path = os.path.join(participant_path, recording)
                    if not os.path.exists(recording_path):
                        continue
                    
                    # Load and process data
                    with open(os.path.join(recording_path, 'labels.json')) as f:
                        labels = json.load(f)
                    
                    acc_data = pd.read_csv(os.path.join(recording_path, 'acceleration.csv'), skiprows=1)
                    acc_data['timestamp'] = (acc_data['timestamp'] - acc_data['timestamp'].iloc[0]) * 1e-9
                    
                    gyro_data = pd.read_csv(os.path.join(recording_path, 'gyroscope.csv'), skiprows=1)
                    gyro_data['timestamp'] = (gyro_data['timestamp'] - gyro_data['timestamp'].iloc[0]) * 1e-9
                    
                    acc_windows, gyro_windows, window_labels = process_recording(
                        acc_data, gyro_data, labels, window_size, stride, flatten
                    )
                    
                    split_acc.extend(acc_windows)
                    split_gyro.extend(gyro_windows)
                    split_labels.extend(window_labels)
                
                # Create dataset for this split if we have data
                if split_acc:
                    # Combine acc and gyro data to the required format
                    if not flatten:
                        # For non-flattened: shape (n_windows, 6, window_size)
                        acc_tensor = torch.tensor(split_acc)
                        gyro_tensor = torch.tensor(split_gyro)
                        X = torch.cat((acc_tensor, gyro_tensor), dim=1)
                    else:
                        # For flattened: shape (n_windows, 2, 3*window_size)
                        acc_tensor = torch.tensor(split_acc)
                        gyro_tensor = torch.tensor(split_gyro)
                        X = torch.stack((acc_tensor, gyro_tensor), dim=1)
                    
                    dataset = MedicationDataset(X, torch.tensor(split_labels), flatten)
                    
                    if split_name == 'train':
                        train_datasets.append(dataset)
                    elif split_name == 'dev':
                        dev_datasets.append(dataset)
                    else:  # test
                        test_datasets.append(dataset)
    
    else:  # LeakageOption.FULL_LEAKAGE
        # Full leakage: collect all data first, then split randomly
        all_X, all_labels = [], []
        
        # Collect data from all participants
        for participant in dir_splits['all_dirs']:
            participant_path = os.path.join(RAW_DATA_DIR, participant)
            X, y = process_participant(participant_path, window_size, stride, flatten)
            
            if len(y) > 0:
                all_X.append(X)
                all_labels.append(y)
        
        # Concatenate all data
        X = torch.cat(all_X) if all_X else torch.tensor([])
        y = torch.cat(all_labels) if all_labels else torch.tensor([])
        
        # Shuffle data
        indices = torch.randperm(len(y))
        X = X[indices]
        y = y[indices]
        
        # Split the data
        n_train = int(len(y) * train_p)
        n_dev = int(len(y) * dev_p)
        
        # Create datasets
        train_dataset = MedicationDataset(
            X[:n_train], 
            y[:n_train],
            flatten
        )
        
        dev_dataset = MedicationDataset(
            X[n_train:n_train + n_dev], 
            y[n_train:n_train + n_dev],
            flatten
        )
        
        test_dataset = MedicationDataset(
            X[n_train + n_dev:], 
            y[n_train + n_dev:],
            flatten
        )
        
        # Package in consistent format
        train_datasets = [train_dataset]
        dev_datasets = [dev_dataset]
        test_datasets = [test_dataset]
    
    # Combine datasets for each split
    train_data = ConcatDataset(train_datasets) if train_datasets else None
    dev_data = ConcatDataset(dev_datasets) if dev_datasets else None
    test_data = ConcatDataset(test_datasets) if test_datasets else None
    
    return {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }


def load_specific_participants(participant_ids, window_size=400, stride=10, flatten=False):
    """
    Load data from specific participants only.
    
    Args:
        participant_ids: List of participant IDs to include
        window_size: Size of the window in samples
        stride: Stride between consecutive windows
        flatten: Whether to flatten x,y,z channels
        
    Returns:
        Dictionary with the dataset
    """
    return load_medication_data(
        window_size=window_size,
        stride=stride,
        flatten=flatten,
        specific_participants=participant_ids
    )


if __name__ == "__main__":
    # Example usage
    data = load_medication_data(window_size=400, stride=10, flatten=False)
    print(f"Training set size: {len(data['train'])}")
    print(f"Dev set size: {len(data['dev'])}")
    print(f"Test set size: {len(data['test'])}")
    
    # Demonstrate data shapes
    train_dataset = data['train'].datasets[0]
    sample_X, sample_y = train_dataset[0]
    
    if not train_dataset.flatten:
        print(f"Non-flattened data shape: {train_dataset.X.shape}")
        print(f"- Expected: (n_windows, 6, window_size)")
        print(f"- Sample X shape: {sample_X.shape}")
    else:
        print(f"Flattened data shape: {train_dataset.X.shape}")
        print(f"- Expected: (n_windows, 2, 3*window_size)")
        print(f"- Sample X shape: {sample_X.shape}")