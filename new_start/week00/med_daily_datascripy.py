import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

SAVE_DIR = '/home/kuba/Documents/data/raw/listerine/transformed_4sec_windows_med_and_daily'

"""For the mediation gestures"""
RAW_DATA_DIR = '/home/kuba/Documents/data/raw/listerine/3_final' 

"""For the daily living gestures"""
DAILY_DIR_PATH = "/home/kuba/Documents/data/raw/kuba_watch_data"


"""The size of each windo in seconds will be window_size/100 """
WINDOW_SIZE = 400 
STRIDE = 10
FLATTEN = False
EXCLUDE_NO_MOVMENT_WINDOWS = True #for the daily living gestures

TRAIN_P = 0.8
DEV_P = 0.1
TEST_P = 0.1

LEAKAGE_ACROSS_PARTICIPANTS = True #one participtes data might be in train and test
LEAKAGE_ACROSS_RECORDINGS = True #if diffrent samples from the same participets racoding are allowed to leak

MEDICINE_CLASS_LABEL = 0 
DAILY_CLASS_LABEL = 1 

#this is for the medication script
labels_to_one_hot = {
    'leftWater' : [MEDICINE_CLASS_LABEL],
    'leftLister' : [MEDICINE_CLASS_LABEL],
    'rightWater' : [MEDICINE_CLASS_LABEL],
    'rightLister' : [MEDICINE_CLASS_LABEL]

}

label_mapping = [
    ('left', 'water', 'leftWater'),
    ('left', 'listerine', 'leftLister'),
    ('right', 'water', 'rightWater'),
    ('right', 'listerine', 'rightLister')
]

#get the med into windows and tensors first

def get_first_line(path):
    f = open(path)
    first_line = int(f.readline().strip().split(':')[1])
    return first_line

def window_maker(data):
    #flatten (bool): If True it combines x,y,z data into single list
    res = []
    if FLATTEN:
        # make windows
        for i in range(0, len(data['x'].tolist()) - WINDOW_SIZE + 1, STRIDE):
            combined = []
            combined.extend(data['x'][i:i + WINDOW_SIZE].tolist())
            combined.extend(data['y'][i:i + WINDOW_SIZE].tolist())
            combined.extend(data['z'][i:i + WINDOW_SIZE].tolist())
            res.append(combined)
    else:
        for i in range(0, len(data['x'].tolist()) - WINDOW_SIZE + 1, STRIDE):
            combined = []
            combined.append(data['x'][i:i + WINDOW_SIZE].tolist())
            combined.append(data['y'][i:i + WINDOW_SIZE].tolist())
            combined.append(data['z'][i:i + WINDOW_SIZE].tolist())
            res.append(combined)
    return res


def recording_accumulator(acc, gyro, labels, window_size, stride, flatten):
    # Splits accelerometer and gyroscope data into segments based on activity labels
    # Returns lists of these data segments and their labels
    acc_bouts, gyro_bouts, y = [], [], []
    
    for side, liquid, label_key in label_mapping:
        if side in labels and liquid in labels[side]:
            for label in labels[side][liquid]:
                new_acc = acc[(acc.timestamp > label['start']) & (acc.timestamp < label['end'])]
                new_gyro = gyro[(gyro.timestamp > label['start']) & (gyro.timestamp < label['end'])]  

                if len(new_acc) >= window_size: #check if we can even get one window out of the bout of activity
                    bouts_windows_acc = window_maker(new_acc, window_size, stride, flatten)#all the windows for a given bout
                    bouts_windows_gyro = window_maker(new_gyro, window_size, stride, flatten)

                    if len(bouts_windows_gyro) != len(bouts_windows_acc):
                        #print(f'ERROR: gyro {len(bouts_windows_gyro)} acc {len(bouts_windows_acc)}')
                        bouts_windows_gyro = bouts_windows_gyro[:-1]
                        #print(f'fixed so: gyro {len(bouts_windows_gyro)} acc {len(bouts_windows_acc)}')


                    acc_bouts.extend(bouts_windows_acc)
                    gyro_bouts.extend(bouts_windows_gyro)
                
                    y.extend([labels_to_one_hot[label_key]] * len(bouts_windows_acc))
    
    return acc_bouts, gyro_bouts, y

def participant_accumulator(dir, window_size, stride, flatten):
    acc_full = []
    gyro_full = []
    y_full = []

    for recoding in sorted(os.listdir(dir)):
        if recoding != '.DS_Store':
            full_path = os.path.join(dir,recoding)
            f = open(os.path.join(full_path,'labels.json'))
            labels = json.load(f)
            acc = pd.read_csv(os.path.join(full_path,'acceleration.csv'), skiprows=1)
            acc['timestamp']  = (acc['timestamp'] - acc['timestamp'].iloc[0]) * 1e-9 #subtract the start to get first time to be zero then convert from nano to sec
            #first_row_acc = get_first_line(os.path.join(full_path, 'acceleration.csv'))

            gyro = pd.read_csv(os.path.join(full_path,'gyroscope.csv'), skiprows=1)
            gyro['timestamp']  = (gyro['timestamp'] - gyro['timestamp'].iloc[0]) * 1e-9 #subtract the start to get first time to be zero then convert from nano to sec
            #first_row_gyro = get_first_line(os.path.join(full_path, 'gyroscope.csv'))
            # print(labels)


            acc, gyro, y = recording_accumulator(acc, gyro, labels, window_size, stride, flatten)
            acc_full.extend(acc)
            gyro_full.extend(gyro)
            y_full.extend(y)
        
    return acc_full, gyro_full, y_full

def get_data_combined():
        train_dirs, dev_dirs, test_dirs = get_directory_splits()
        train_datasets = []
        test_datasets = []

        for dir in sorted(os.listdir(raw_data_dir)):
            if dir != '.DS_Store':
                X_acc, X_gyro, y = participant_accumulator(os.path.join(raw_data_dir, dir), window_size, stride, flatten)
                # Create datasets without internal shuffling
                dataset = CombinedDataSet(X_acc=X_acc, X_gyro=X_gyro, y=y, 
                                        seed=None, flatten=flatten)  # we remove  internal shuffle since we suffle across th datast at the end
                
                if dir in train_sets:
                    train_datasets.append(dataset)
                else:
                    test_datasets.append(dataset)

        # Combine datasets
        train_data = ConcatDataset(train_datasets)
        test_data = ConcatDataset(test_datasets)

        # Single shuffle of combined datasets
        train_indices = torch.randperm(len(train_data))
        test_indices = torch.randperm(len(test_data))
        
        train_data = [train_data[i] for i in train_indices]
        test_data = [test_data[i] for i in test_indices]
        