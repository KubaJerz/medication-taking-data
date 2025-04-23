import os
import pandas as pd
import json
import numpy as np

# Label mappings
LABEL_MAPPING = [
    ('left', 'water', 'leftWater'),
    ('left', 'listerine', 'leftLister'),
    ('right', 'water', 'rightWater'),
    ('right', 'listerine', 'rightLister')
]

def get_bouts_from_recording(recording_dir, min_bout_len):
    acc_data = pd.read_csv(os.path.join(recording_dir, 'acceleration.csv'), skiprows=1)
    acc_data['timestamp'] = (acc_data['timestamp'] - acc_data['timestamp'].iloc[0]) * 1e-9
    gyro_data = pd.read_csv(os.path.join(recording_dir, 'gyroscope.csv'), skiprows=1)
    gyro_data['timestamp'] = (gyro_data['timestamp'] - gyro_data['timestamp'].iloc[0]) * 1e-9
    
    with open(os.path.join(recording_dir, 'labels.json')) as f:
        labels = json.load(f)
    
    combined_bouts, labels_list = [], []
    
    for side, liquid, label_key in LABEL_MAPPING:
        if side in labels and liquid in labels[side]:
            for label in labels[side][liquid]:
                # Filter data within the labeled time range
                filtered_acc = acc_data[(acc_data.timestamp > label['start']) & (acc_data.timestamp < label['end'])]
                filtered_gyro = gyro_data[(gyro_data.timestamp > label['start']) & (gyro_data.timestamp < label['end'])]
                
                # Extract just the x, y, z columns
                acc_bout = filtered_acc[['x', 'y', 'z']].values
                gyro_bout = filtered_gyro[['x', 'y', 'z']].values
                
                # Handle different lengths by using the smaller length
                min_length = min(len(acc_bout), len(gyro_bout))
                if min_length < min_bout_len:
                    continue #do not add about that are less than the min len

                acc_bout = acc_bout[:min_length]
                gyro_bout = gyro_bout[:min_length]
                
                # Stack acc and gyro data to get shape [6, len_of_bout]
                # acc_bout and gyro_bout both have shape [len_of_bout, 3]
                # We need to transpose them to [3, len_of_bout] and then stack
                combined_bout = np.vstack([acc_bout.T, gyro_bout.T])  # Shape: [6, len_of_bout]
                
                combined_bouts.append(combined_bout)
                labels_list.append(label_key)
    
    return combined_bouts, labels_list

def get_participant_bouts(path_to_single_participants_dir, min_bout_len):
    all_bouts, all_labels = [], []
    
    for recording_dir in sorted(os.listdir(path_to_single_participants_dir)):
        if recording_dir == '.DS_Store':
            continue
            
        bouts, labels = get_bouts_from_recording(os.path.join(path_to_single_participants_dir, recording_dir), min_bout_len)
        
        all_bouts.extend(bouts)
        all_labels.extend(labels)
    
    return all_bouts, all_labels

def get_bouts(path_to_all_participants_dir, min_bout_len=0, shuffle=True, who_no_to_sample_from = []):
    print(f"WARNING we will sample bouts from all participants there will be data leakage, unless you specify who not to sample from")
    
    all_bouts, all_labels = [], []
    
    for participant in sorted(os.listdir(path_to_all_participants_dir)):
        if participant in who_no_to_sample_from or participant == '.DS_Store':
            continue
            
        bouts, labels = get_participant_bouts(os.path.join(path_to_all_participants_dir, participant), min_bout_len=min_bout_len)
        
        all_bouts.extend(bouts)
        all_labels.extend(labels)

    #convert to numpy array
    all_labels = np.array(all_labels)

    if shuffle:
        np.random.seed(69)
        indices = np.random.permutation(len(all_bouts))
        all_bouts = [all_bouts[i] for i in indices]
        all_labels = all_labels[indices]
    
    return all_bouts, all_labels