import pandas as pd
import os 
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_data(path_to_dir):
    """
    Returns as a pandas DF
    """
    #accl
    accl_path = os.path.join(path_to_dir, "acceleration.csv")
    #gyro
    gyro_path = os.path.join(path_to_dir, "gyroscope.csv")

    acc = pd.read_csv(accl_path, skiprows=1)
    acc['timestamp']  = (acc['timestamp'] - acc['timestamp'].iloc[0]) * 1e-9 #subtract the start to get first time to be zero then convert from nano to sec
    #first_row_acc = get_first_line(os.path.join(full_path, 'acceleration.csv'))

    gyro = pd.read_csv(gyro_path, skiprows=1)
    gyro['timestamp']  = (gyro['timestamp'] - gyro['timestamp'].iloc[0]) * 1e-9 #subtract the start to get first time to be zero then convert from nano to sec
    #first_row_gyro = get_first_line(os.path.join(full_path, 'gyroscope.csv'))
    return acc, gyro


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


def process_recording(acc_data, gyro_data, window_size, stride, flatten):
    """Extract activity windows and labels from a recording."""

    if len(acc_data) >= window_size:
        acc_windows_batch = window_maker(acc_data, window_size, stride, flatten)
        gyro_windows_batch = window_maker(gyro_data, window_size, stride, flatten)

        # Handle potential mismatch in window counts
        min_windows = min(len(acc_windows_batch), len(gyro_windows_batch))
        acc_windows_batch = acc_windows_batch[:min_windows]
        gyro_windows_batch = gyro_windows_batch[:min_windows]
    
    return torch.tensor(acc_windows_batch), torch.tensor(gyro_windows_batch)

def smooth_predictions(prediction_sum, counts):
    """
    Smooth predictions by avg handling divisions by zero
    """

    averaged_predictions = prediction_sum / (counts + 0.00001)
    return averaged_predictions

def plot_full_preds(x, y, z, avg_preds, confidance_threshold_for_mean, scale_rate=50):
    fig, ax = plt.subplots(figsize=(18, 6))

    ax.plot(x, label='x')
    ax.plot(y, label='y')
    ax.plot(z, label='z')
    ax.plot((avg_preds * scale_rate), label=f'Rolling avg confidence of being medication gesture for 0-{scale_rate}')
    ax.hlines([(confidance_threshold_for_mean * scale_rate)],xmin=0, xmax=len(x), linestyles='dashdot', label='Threshold')
    ax.set_ylabel('Accelation x,y,z')
    ax.set_xlabel('Time in centisecond (0.01 sec)')
    ax.grid()
    ax.legend()

    plt.show()

def count_medication_taking_predictions(avg_preds, window_size, stride, conf_threshold):
    count = 0
    total_num_windows = 0
    for i in range(0, len(avg_preds) - window_size, stride):
        total_num_windows += 1
        window = avg_preds[i: i + window_size]
        if np.mean(window) >= conf_threshold :
            count += 1

    return count, total_num_windows    


def get_false_postive_count(model, path_to_dir, window_size, stride, confidance_threshold_for_mean=0.8, device='cpu', flatten=False):
    acc, gyro = get_data(path_to_dir)
    acc_windows, gyro_windows = process_recording(acc, gyro, window_size=window_size, stride=stride, flatten=flatten)
    all_windows = torch.cat((acc_windows, gyro_windows), dim=1)
    print(f"The recording partitioned into windows has shape of: {all_windows.shape}")

    #get preds
    model = model.to(device)
    all_windows = all_windows.to(device)
    with torch.no_grad():
        model.eval()
        logits = model(all_windows)

    #process the preds
    preds = torch.nn.functional.sigmoid(logits.cpu())
    preds_list = preds.tolist()
    preds_list = [val[0] for val in preds_list]
    preds_list.reverse() # Reverse list once then use .pop() instead of .pop(0) cuz faster
                         # .pop() is O(1) while .pop(0) requires shifting all elements left (O(n))
    
    len_to_use = min(len(acc) - window_size, len(gyro) - window_size) #Handle potential mismatch in window count
    counts = np.zeros((len(acc)))
    sums = np.zeros((len(acc)))

    #create the rolling window sums
    for i in range(0, len_to_use, stride):
        val = preds_list.pop()
        sums[i : i+window_size] += val
        counts[i : i+window_size] += 1

    avg_preds = smooth_predictions(sums, counts)

    x, y, z = acc['x'].to_numpy(), acc['y'].to_numpy(), acc['z'].to_numpy()
    plot_full_preds(x, y, z, avg_preds, confidance_threshold_for_mean)

    count, total_num_windows = count_medication_taking_predictions(avg_preds, window_size=window_size, stride=window_size, conf_threshold=confidance_threshold_for_mean)
    print(f"There is {count} of {total_num_windows} predictions over the confidance threshold({confidance_threshold_for_mean})")
    return count, total_num_windows

