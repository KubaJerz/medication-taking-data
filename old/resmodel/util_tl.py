import os
import sys
import json
import torch
import importlib
import matplotlib.pyplot as plt
import glob
import json
import numpy as np



def plot_combined_metrics(lossi, devlossi, f1i, devf1i, best_f1_dev, best_loss_dev, train_id, save_dir):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lossi, label='Train Loss')
    plt.plot(devlossi, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.axhline(best_loss_dev, color='g', linestyle='--', label=f'Best Dev Loss: {best_loss_dev:.3f}')
    #plt.text(0, best_loss_dev - 0.02, f'Best Loss: {best_loss_dev:.4f}', color='g', fontsize=10)
    plt.legend()
    
    # Plot F1 scores
    plt.subplot(1, 2, 2)
    plt.plot(f1i, label='Train F1')
    plt.plot(devf1i, label='Validation F1')
    plt.title('F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.axhline(best_f1_dev, color='g', linestyle='--', label=f'Best Dev F1: {best_f1_dev:.3f}')
    #plt.text(0, best_f1_dev - 0.02, f'Best F1: {best_f1_dev:.4f}', color='g', fontsize=10)
    plt.legend()

    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"{train_id}_metrics.png"))
    plt.close()


def save_metrics(metrics, filename):
    metrics['confusion_matrix'] = metrics['confusion_matrix'].tolist() if isinstance(metrics['confusion_matrix'],np.ndarray) else metrics['confusion_matrix']
    with open(filename, 'w') as f:
        json.dump(metrics, f)


def extract_metrics(directory,full, f1, loss):
    metrics_data = []
    if full:
        # Search for files matching the pattern *_metrics.json 
        file_pattern = os.path.join(directory, '*_Full_metrics.json')
        files = glob.glob(file_pattern)
    if f1: #we are resuming from the best f1 score of the model
        file_pattern = os.path.join(directory, '*_bestF1_metrics.json')
        files = glob.glob(file_pattern)
    if loss: #we are resuming from the best loss score of the model
        file_pattern = os.path.join(directory, '*_bestLoss_metrics.json')
        files = glob.glob(file_pattern)

    if not files:
        print(f"No previous metrics files found in {directory}")
        return {}
    
    file_path = files[0]

    try:
        with open(file_path, 'r') as file:
            metrics = json.load(file)
        return metrics
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return {}

def check_resume(model_path, f1, full, loss):
    if f1 and ('f1' not in model_path.lower()):
        res_type = 'f1'
        raise ValueError(f"Model path '{model_path}' must contain '{res_type}' when the {res_type} resume flag is True\nPlease make sure that the model you are resuming matches the resume type")
    if loss and ('loss' not in model_path.lower()):
        res_type = 'loss'
        raise ValueError(f"Model path '{model_path}' must contain '{res_type}' when the {res_type} resume flag is True\nPlease make sure that the model you are resuming matches the resume type")
    if full and ('f1' in model_path.lower() or 'loss' in model_path.lower()):
        res_type = 'loss'
        raise ValueError(f"Model path '{model_path}' must contain '{res_type}' when the {res_type} resume flag is True\nPlease make sure that the model you are resuming matches the resume type")


def check_and_save_best_model(metrics, model, epoch, save_dir, train_id, best_dev_loss, best_dev_f1):
    is_loss_best = False
    is_f1_best = False

    if metrics['dev_loss'][-1] < best_dev_loss:
        is_loss_best = True
        best_dev_loss = metrics['dev_loss'][-1]
        metrics['best_loss_dev'] = best_dev_loss

    if metrics['dev_f1'][-1] > best_dev_f1:
        is_f1_best = True
        best_dev_f1 = metrics['dev_f1'][-1]
        metrics['best_f1_dev'] = best_dev_f1

    if is_f1_best or is_loss_best:
        print('\n')

        if is_loss_best:
            model_path = os.path.join(save_dir, f'{train_id}_bestLoss.pth')
            metrics_path = os.path.join(save_dir, f'{train_id}_bestLoss_metrics.json')
            if os.path.exists(model_path):
                os.remove(model_path)
                os.remove(metrics_path)
            torch.save(model, model_path)
            save_metrics(metrics, metrics_path)
            print(f'New best Loss: {best_dev_loss} at Epoch:{epoch} Model saved!')

        if is_f1_best:
            model_path = os.path.join(save_dir, f'{train_id}_bestF1.pth')
            metrics_path = os.path.join(save_dir, f'{train_id}_bestF1_metrics.json')
            if os.path.exists(model_path):
                os.remove(model_path)
                os.remove(metrics_path)
            torch.save(model, model_path)
            save_metrics(metrics, metrics_path)
            print(f'New best F1: {best_dev_f1} at Epoch:{epoch} Model saved!')

        print('\n')

    return best_dev_loss, best_dev_f1

def resume_training(model_path, training_id, f1, full, loss):
    print(f"We are resuming training of model found at: {model_path}\n\n")
    model = torch.load(model_path)
    dir_path, file_name = os.path.split(model_path)
    check_resume(model_path=model_path, f1=f1, full=full, loss=loss)
    model_name = (file_name.split('.')[0]).upper()
    metrics = extract_metrics(dir_path, full=full, f1=f1, loss=loss)
    save_dir = os.path.join('.', f'{training_id}')
    os.makedirs(save_dir, exist_ok=True)
    ttx_file_path = os.path.join(save_dir, "desc.txt")
    with open(ttx_file_path, 'w') as ttx_file:
        json.dump(f'model that was resumed was named {model_name} located at {model_path}', ttx_file, indent=4)

    return model, metrics, save_dir, model_name

def new_training(model_path, training_id):
    print(f"We are training a NEW model defined at: {model_path}\n\n")
    dir_path, file_name = os.path.split(model_path)
    module_name = (file_name.split('.')[0])
    model_name = (file_name.split('.')[0]).upper()
    metrics = {
        'train_loss': [], 'dev_loss': [], 'train_f1': [], 'dev_f1': [],
        'best_f1_dev': 0, 'best_loss_dev': float('inf'), 'confusion_matrix': None
    }
    sys.path.append(dir_path)
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_name)
    model = model_class()
    save_dir = os.path.join('.', f'{training_id}')
    os.makedirs(save_dir, exist_ok=True)
    ttx_file_path = os.path.join(save_dir, "desc.txt")
    with open(ttx_file_path, 'w') as ttx_file:
        json.dump(f'model that was trained was named {model_name} located at {model_path}', ttx_file, indent=4)
    return model, metrics, save_dir, model_name

def model_instance_training(model_instance, sub_dir):
    metrics = {
        'train_loss': [], 'dev_loss': [], 'train_f1': [], 'dev_f1': [],
        'best_f1_dev': 0, 'best_loss_dev': float('inf'), 'confusion_matrix': None
    }
    model = model_instance
    save_dir = sub_dir
    os.makedirs(save_dir, exist_ok=True)
    return model, metrics, save_dir, 'noName'

def train_prep(model_input, resume=False, f1=False, full=False, loss=False, training_id=None, sub_dir=None):
    if isinstance(model_input, str):
        model_path = os.path.abspath(model_input)
        if resume:
            return resume_training(model_path=model_path, training_id=training_id, f1=f1, full=full, loss=loss)
        else:
            return new_training(model_path=model_path, training_id=training_id)
    else:
        return model_instance_training(model_instance=model_input, sub_dir=sub_dir)

# You might need to define or import these functions:
# check_resume()
# extract_metrics()