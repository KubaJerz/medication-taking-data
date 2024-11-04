'''
Read the docs for a better explination but here theres are much more customizable to each solution
So the path and stuff is hard coded we will only call the get sata set method with a radnom state

how to use:

from 0_tensor_builder.py import getDataSet
train_dataset, test_dataset = getDataSet()

'''
import torch 
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import random
import os 
# import medication_data_maker

raw_data_dir = '/home/kuba/Documents/Data/Raw/Listerine/3_final'

#if 4 classes
labels_to_one_hot = {
    'leftWater' : [[0.0],[0.0],[0.0],[1.0]],
    'leftLister' : [[0.0],[0.0],[1.0],[0.0]],
    'rightWater' : [[0.0],[1.0],[0.0],[0.0]],
    'rightLister' : [[1.0],[0.0],[0.0],[0.0]]

}


#if 2 classes
labels_to_one_hot = {
    'leftWater' : [0.0],
    'leftLister' : [1.0],
    'rightWater' : [0.0],
    'rightLister' : [1.0]

}

label_mapping = [
    ('left', 'water', 'leftWater'),
    ('left', 'listerine', 'leftLister'),
    ('right', 'water', 'rightWater'),
    ('right', 'listerine', 'rightLister')
]


class CombinedDataSet(Dataset):
    def __init__(self, X_acc, X_gyro, y, seed=None, flatten=True):
        # Shuffle the data within each set
        if flatten:
            # Concatenate along the last dimension to get torch.Size([669, 1800])
            self.x = torch.cat((X_acc, X_gyro), dim=1)
        else:
            # Stack along a new dimension if flatten is False
            self.x = torch.stack((X_acc, X_gyro), dim=1)  # Resulting shape will be torch.Size([669, 2, 900])

        self.y = y

        if seed is not None:
            # Set the seed for reproducibility
            random.seed(seed)
            torch.manual_seed(seed)
            
            # Shuffle indices
            indices = torch.randperm(len(self.y))
            self.x = self.x[indices]
            self.y = self.y[indices]
        
        self.n_samples = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def window_maker(data, window_size, stride, flatten):
    #flatten (bool): If True it combines x,y,z data into single list
    res = []
    if flatten:
        # make windows
        for i in range(0, len(data['x'].tolist()) - window_size + 1, stride):
            combined = []
            combined.extend(data['x'][i:i + window_size].tolist())
            combined.extend(data['y'][i:i + window_size].tolist())
            combined.extend(data['z'][i:i + window_size].tolist())
            res.append(combined)
    else:
        for i in range(0, len(data['x'].tolist()) - window_size + 1, stride):
            combined = []
            combined.append(data['x'][i:i + window_size].tolist())
            combined.append(data['y'][i:i + window_size].tolist())
            combined.append(data['z'][i:i + window_size].tolist())
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

                    if len(bouts_windows_gyro) > len(bouts_windows_acc):
                        bouts_windows_gyro = bouts_windows_gyro[:len(bouts_windows_acc)]
                    elif len(bouts_windows_acc) > len(bouts_windows_gyro):
                        bouts_windows_acc = bouts_windows_acc[:len(bouts_windows_gyro)]


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
        
    return torch.tensor(acc_full), torch.tensor(gyro_full), torch.tensor(y_full)


def getDataSet(randomState=69, train_sets=['00','01','02','03','04','05','06','07','08','09','10','11'], window_size=300, stride=50, flatten=True):
    train_datasets = []
    test_datasets = []


    for dir in sorted(os.listdir(raw_data_dir)):
        if dir != '.DS_Store':
            X_acc, X_gyro, y = participant_accumulator(os.path.join(raw_data_dir, dir), window_size, stride, flatten)
            # Shuffle within each dataset and add to train or test
            if dir in train_sets:
                train_datasets.append(CombinedDataSet(X_acc=X_acc, X_gyro=X_gyro, y=y, seed=randomState, flatten=flatten))
            else:
                test_datasets.append(CombinedDataSet(X_acc=X_acc, X_gyro=X_gyro, y=y, seed=randomState, flatten=flatten))

    # Concatenate and shuffle the combined train and test datasets
    train_data = ConcatDataset(train_datasets)
    test_data = ConcatDataset(test_datasets)

    # Shuffle combined datasets
    def shuffle_dataset(dataset, seed):
        indices = torch.randperm(len(dataset))
        dataset = [dataset[i] for i in indices]
        return dataset

    # Apply shuffling on the combined datasets
    train_data = shuffle_dataset(train_data, randomState)
    test_data = shuffle_dataset(test_data, randomState)

    return train_data, test_data

if __name__ == "__main__":
    train_dataset, test_dataset = getDataSet()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
