import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from enum import Enum
from typing import List, Dict, Optional, Tuple

# Import the data loaders from their respective modules
from medication_data_script import load_medication_data, load_specific_participants, LeakageOption
from daily_data_scripy import load_daily_data

def get_data(window_size=400, 
            stride=10, 
            flatten=False,
            train_p=0.8, 
            dev_p=0.1, 
            test_p=0.1,
            leakage_option=LeakageOption.NO_LEAKAGE,
            specific_participants=None,
            filter_static=True):
    """
    Get both medication and daily living gesture data, with balanced classes.
    
    Args:
        window_size: Size of the window (in samples)
        stride: Stride between consecutive windows
        flatten: Whether to flatten x,y,z channels
        train_p: Proportion for training set
        dev_p: Proportion for dev/validation set  
        test_p: Proportion for test set
        leakage_option: How to handle data leakage between splits
        specific_participants: List of specific participant IDs to include (optional)
        balance_classes: Whether to balance the classes to have equal numbers
        filter_static: Whether to filter out windows with little movement in daily data
        
    Returns:
        Dictionary with train, dev, and test datasets
    """
    # For specific participants, just return medication data
    if specific_participants:
        specific_data = load_specific_participants(
            participant_ids=specific_participants,
            window_size=window_size,
            stride=stride,
            flatten=flatten
        )
        return {
            'train': specific_data['train'],
            'dev': specific_data['dev'],
            'test': specific_data['test'],
            'data_type': 'medication_only'
        }
    
    # Load medication data (class 0)
    med_data = load_medication_data(
        window_size=window_size,
        stride=stride,
        flatten=flatten,
        train_p=train_p,
        dev_p=dev_p,
        test_p=test_p,
        leakage_option=leakage_option
    )
    
    # Get dataset sizes for each split
    train_med_size = len(med_data['train']) if med_data['train'] else 0
    dev_med_size = len(med_data['dev']) if med_data['dev'] else 0
    test_med_size = len(med_data['test']) if med_data['test'] else 0
    
    # Load daily living data (class 1) with matching sizes
    daily_data = load_daily_data(
        num_train_samples=train_med_size,
        num_dev_samples=dev_med_size,
        num_test_samples=test_med_size,
        window_size=window_size,
        stride=stride,
        flatten=flatten,
        filter_static=filter_static
    )
    
    # # Balance classes if requested
    # if balance_classes:
    #     train_data, dev_data, test_data = balance_datasets(med_data, daily_data)
    # else:
    #     # Combine datasets for each split
    train_data = ConcatDataset([med_data['train'], daily_data['train']]) if med_data['train'] and daily_data['train'] else None
    dev_data = ConcatDataset([med_data['dev'], daily_data['dev']]) if med_data['dev'] and daily_data['dev'] else None
    test_data = ConcatDataset([med_data['test'], daily_data['test']]) if med_data['test'] and daily_data['test'] else None

    # Return combined datasets
    return {
        'train': train_data,
        'dev': dev_data,
        'test': test_data,
        'data_type': 'combined'
    }


# def balance_datasets(med_data, daily_data):
#     """
#     Balance the medication and daily datasets to have equal numbers of samples in each class.
    
#     Args:
#         med_data: Dictionary with medication datasets
#         daily_data: Dictionary with daily living datasets
        
#     Returns:
#         Tuple of (train_dataset, dev_dataset, test_dataset)
#     """
#     # Balance training set
#     train_balanced = balance_split(med_data['train'], daily_data['train'])
    
#     # Balance dev set
#     dev_balanced = balance_split(med_data['dev'], daily_data['dev'])
    
#     # Balance test set
#     test_balanced = balance_split(med_data['test'], daily_data['test'])
    
#     return train_balanced, dev_balanced, test_balanced


# def balance_split(med_dataset, daily_dataset):
#     """
#     Balance a single dataset split to have equal numbers from each class.
    
#     Args:
#         med_dataset: Medication dataset
#         daily_dataset: Daily living dataset
        
#     Returns:
#         Balanced combined dataset
#     """
#     # Handle cases where one dataset might be None
#     if med_dataset is None:
#         return daily_dataset
#     if daily_dataset is None:
#         return med_dataset
    
#     # Get dataset sizes
#     med_size = len(med_dataset)
#     daily_size = len(daily_dataset)
    
#     # Get the size of the smaller dataset
#     min_size = min(med_size, daily_size)
    
#     # Sample from the larger dataset if needed
#     if med_size > min_size:
#         # Sample random indices without replacement
#         indices = torch.randperm(med_size)[:min_size]
#         med_dataset_balanced = Subset(med_dataset, indices)
#         daily_dataset_balanced = daily_dataset
#     elif daily_size > min_size:
#         indices = torch.randperm(daily_size)[:min_size]
#         daily_dataset_balanced = Subset(daily_dataset, indices)
#         med_dataset_balanced = med_dataset
#     else:
#         # Already balanced
#         med_dataset_balanced = med_dataset
#         daily_dataset_balanced = daily_dataset
    
#     # Combine the balanced datasets
#     combined_dataset = ConcatDataset([med_dataset_balanced, daily_dataset_balanced])
    
#     return combined_dataset


if __name__ == "__main__":
    # Example usage
    data = get_data(
        window_size=400, 
        stride=100,
        flatten=False,
        leakage_option=LeakageOption.NO_LEAKAGE,
        balance_classes=True
    )
    
    print(f"Training set size: {len(data['train'])}")
    print(f"Dev set size: {len(data['dev'])}")
    print(f"Test set size: {len(data['test'])}")
    
    # Check data distribution by looking at the class labels
    def check_class_balance(dataset):
        labels = []
        for i in range(min(100, len(dataset))):  # Check first 100 samples
            _, label = dataset[i]
            labels.append(label.item())
        
        class_0_count = labels.count(0)
        class_1_count = labels.count(1)
        
        print(f"Class 0 (Medication): {class_0_count} samples")
        print(f"Class 1 (Daily): {class_1_count} samples")
        print(f"Ratio: {class_0_count / (class_0_count + class_1_count):.2f}")
    
    print("\nTraining set class distribution:")
    check_class_balance(data['train'])