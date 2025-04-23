import pandas as pd
import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from medication_bout_script import get_bouts

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

def get_random_med_bouts(all_bouts, num_bouts_to_samp):
    bout_picks = []
    idxs_used = []

    for _ in range(num_bouts_to_samp):
        pick = random.randint(0, len(all_bouts) - 1)
        while pick in idxs_used:
            pick = random.randint(0, len(all_bouts) - 1)
        
        bout_picks.append(all_bouts[pick]) #add window
        idxs_used.append(pick) #add so we dont reuse same windows

    return bout_picks
    
def paste_in_bouts(bouts, acc, gyro):

    def is_vaild(i, bout_len, idxs, full_length):
        if i < 0:
            return False
        elif (i + bout_len) > full_length:
            return False
        else:
            for pair in idxs:
                j, j_plus_w = pair #pais of windows we have already pasted
                if ((i + bout_len) >= j) and ((i + bout_len) <=  j_plus_w):
                    return False
                elif ((i) >= j) and ((i) <=  j_plus_w):
                    return False
            return True

    acc_augmented = acc
    gyro_augmented = gyro
    idxs = [] #list of tuples (i, i+bout_len) where each bout is placed
    for bout in bouts:
        bout_acc = bout[:3,:]
        bout_gyro = bout[3:,:]
        print(f"Bout shape - Acc: {len(bout_acc[0])}, Gyro: {len(bout_gyro[0])}")

        valid_placment = False
        while not valid_placment:
            i = random.randint(0, len(acc) -1)
            valid_placment = is_vaild(i,len(bout[0]), idxs, len(acc))

            if valid_placment:
                acc_augmented[i:i+len(bout_acc[0])] = bout_acc.T
                gyro_augmented[i:i+len(bout_acc[0])] = bout_gyro.T
                idxs.append((i, i+len(bout_acc[0]))) 
    
    return acc_augmented, gyro_augmented, idxs 

def window_maker(data, window_size, stride, flatten):
    """Create windows from time series data."""
    windows = []
    
    if flatten:
        # Flatten x, y, z channels into a single vector per window
        for i in range(0, len(data[:,0]) - window_size + 1, stride):
            window = []
            window.extend(data[:,0][i:i + window_size].tolist())
            window.extend(data[:,1][i:i + window_size].tolist())
            window.extend(data[:,2][i:i + window_size].tolist())
            windows.append(window)
    else:
        # Keep x, y, z as separate channels
        for i in range(0, len(data[:,0]) - window_size + 1, stride):
            window = [
                data[:,0][i:i + window_size].tolist(),
                data[:,1][i:i + window_size].tolist(),
                data[:,2][i:i + window_size].tolist()
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


def plot_full_preds(x, y, z, avg_preds, idxs, confidance_threshold_for_mean, scale_rate=50):
    fig, ax = plt.subplots(figsize=(24, 10))

    ax.plot(x, label='x')
    ax.plot(y, label='y')
    ax.plot(z, label='z')
    ax.plot((avg_preds * scale_rate), label=f'Rolling avg confidence of being medication gesture for 0-{scale_rate}')
    ax.hlines([(confidance_threshold_for_mean * scale_rate)],xmin=0, xmax=len(x), linestyles='dashdot', label='Threshold')

    for idx_pair in idxs:
        start_idx, end_idx = idx_pair
        plt.axvspan(start_idx, end_idx, color='darkgray', alpha=0.6)
        plt.text(start_idx, plt.ylim()[1]*0.9, f"bout", fontsize=4, rotation=90, backgroundcolor='white', alpha=0.7)


    ax.set_ylabel('Accelation')
    ax.set_xlabel('Time in centisecond (0.01 sec)')
    ax.grid()
    ax.legend()

    plt.show()

def show_cfm(TP, TN, FP, FN, total):
    row0_sum = TN + FP
    row1_sum = FN + TP

    # Normalized by "true" (rows)
    cf_norm_true = np.array([
        [TN / row0_sum, FP / row0_sum],
        [FN / row1_sum, TP / row1_sum]
    ])

    cf_norm_true_disp = ConfusionMatrixDisplay(confusion_matrix=cf_norm_true)
    print("Normalized on True")
    cf_norm_true_disp.plot()

    cf = np.array([[TN, FP],[FN, TP]])
    cf_disp = ConfusionMatrixDisplay(confusion_matrix=cf)
    cf_disp.plot()

def count_peaks_above_threshold(array, threshold):
    if len(array) < 3:  # Need at least 3 points to have a peak
        return 0, []
    
    peak_count = 0
    in_peak = False
    peak_idxs = []
    
    for i in range(1, len(array)):
        # start a peak when avg conf rises above threshold
        if not in_peak and array[i] > threshold and array[i] > array[i-1]:
            in_peak = True
            peak_count += 1
            start = i
            
        # End of peak (value falls below threshold)
        elif in_peak and array[i] <= threshold:
            in_peak = False
            end = i-1
            peak_idxs.append((start, end))
    
    #  case where the array ends and we still in a peak
    if in_peak:
        peak_idxs.append((start, len(array)-1))
        
    return peak_count, peak_idxs

def check_overlapping_peaks(peak_idxs, bout_idxs, overlap_threshold=0.85):
    TP = []  # true positives
    FP = []  # fasle positives
    
    for peak in peak_idxs:
        peak_start, peak_end = peak
        peak_length = peak_end - peak_start + 1
        
        is_overlapping = False
        
        for bout in bout_idxs:
            bout_start, bout_end = bout
            
            overlap_start = max(peak_start, bout_start)
            overlap_end = min(peak_end, bout_end)
            
            if overlap_start <= overlap_end:  
                overlap_length = overlap_end - overlap_start + 1
                
                peak_overlap_ratio = overlap_length / peak_length
                
                if peak_overlap_ratio >= overlap_threshold:                    
                    TP.append(peak)
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            FP.append(peak)
    
    return TP, FP


def eval_in_time_domain(model, path_to_daily_dir, path_to_bout_dir, window_size, stride, confidance_threshold_for_mean=0.8, overlap_threshold=0.45, min_bout_len=0, num_bouts_to_samp=20, device='cpu', flatten=False):
    acc_with_timestamp, gyro_with_timestamp = get_data(path_to_daily_dir)
    all_bouts, _ = get_bouts(path_to_bout_dir, shuffle=True, min_bout_len=min_bout_len)
    sampled_bouts = get_random_med_bouts(all_bouts, num_bouts_to_samp)
    acc = acc_with_timestamp.iloc[:, 1:4].to_numpy()
    gyro = gyro_with_timestamp.iloc[:, 1:4].to_numpy() 
    
    acc_augmented, gyro_augmented, bout_idxs = paste_in_bouts(sampled_bouts, acc, gyro)

    acc_windows, gyro_windows = process_recording(acc_augmented, gyro_augmented, window_size=window_size, stride=stride, flatten=flatten)
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


    x, y, z = acc_augmented[:,0], acc_augmented[:,1], acc_augmented[:,2]
    plot_full_preds(x, y, z, avg_preds, bout_idxs, confidance_threshold_for_mean)

    total_peak_count, peak_idxs = count_peaks_above_threshold(avg_preds, threshold=confidance_threshold_for_mean)
    TP, FP = check_overlapping_peaks(peak_idxs, bout_idxs, overlap_threshold=overlap_threshold)
    FP_count = len(FP)
    TP_count = len(TP)

    print(f"There are {FP_count} FP's of {total_peak_count} total peaks (confidance over: {confidance_threshold_for_mean} and does not overlap with bout threshold:{overlap_threshold})")
    print(f"FP's are at rate of {(FP_count/total_peak_count)*100:.2f}%")
    print(f"TP's are at rate of {(TP_count/total_peak_count)*100:.4f}%")
    print(f"Out of {num_bouts_to_samp} model found {TP_count} ({(TP_count/num_bouts_to_samp)*100:.2f})")

    FN_count = num_bouts_to_samp - TP_count
    TN_count = int(len(acc_augmented / 400) - TP_count - FN_count - FP_count)
    show_cfm(TP_count, TN_count, FP_count, FN_count, total_peak_count)
    
    return TP_count, TN_count, FP_count, FN_count, total_peak_count 




