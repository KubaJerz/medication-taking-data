import random
import argparse
import os
import json

'''MAKE SURE TO UPDATE THESE'''
from resmodel import RESMODEL  #this is the base model that when passed in new hyper params will create new models
from tl06 import run_training


def def_model(hyperparams, experimet_dir, input_channel, out_sz, num_classes):
    sub_dir = os.path.join(experimet_dir,hyperparams['id'])
    os.makedirs(sub_dir,exist_ok=False)

    model = RESMODEL(input_channels=input_channel, output_size=out_sz, hyperparams=hyperparams, num_classes=num_classes)
    
    ttx_file_path = os.path.join(sub_dir, "desc.txt")
    with open(ttx_file_path, 'w') as ttx_file:
        json.dump(hyperparams, ttx_file, indent=4)

    return model, sub_dir

def train_model(hyperparams, model, sub_dir, epochs):
    run_training(training_id=hyperparams['id'],
                 model_path=model,
                 sub_dir=sub_dir,
                 epochs=epochs,
                 window_size=hyperparams['window_size'],
                 stride =hyperparams['stride'],
                 train_batch_size=hyperparams['train_batch_size'],
                 test_batch_size=hyperparams['test_batch_size'],
                 lr=hyperparams['learning_rate'])
    


def main():
    parser = argparse.ArgumentParser(description="train wrapper")
    parser.add_argument("id", type=str, help="ID")
    parser.add_argument("--num_models", type=int, default=5, help="Number of models to train (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default: 50)")


    args = parser.parse_args()
    id = args.id
    num_models = args.num_models
    epochs = args.epochs

    '''________________ADJUST THESES___________________'''
    input_channel = 6
    output_sz = 1
    num_classes = 2

    hyper_ranges = {
        'window_size': (2.5,4.5),
        'stride': (1,50),
        'train_batch_size': [32,64,128,256,512],
        'learning_rate': (0.0001, 0.1),
        'hidden_blocks': [1,2,3,4,5],
        'layer_zero_depth': [1,2,3],
        'layer_scaler': [1,2,3],
        'dropout_rate': (0.1, 0.5),
        'activation': ['ReLU'], #['ReLU', 'tanh', 'sigmoid'],
        'normalization': ['batch'] #, 'layer']#['none', 'batch', 'layer']
    }
    test_batch_size = -1

    '''____________________________________________'''

    experimet_dir = os.path.join(os.getcwd(),f'{id}')

    for i in range(num_models):
        #random pick
        hyperparams = {
            'id': f"{id}_{i}",
            'window_size': int(random.uniform(*hyper_ranges['window_size']) * 100),
            'stride': int(random.uniform(*hyper_ranges['stride'])),
            'train_batch_size': random.choice(hyper_ranges['train_batch_size']),
            'test_batch_size': test_batch_size,
            'epochs': epochs,
            'learning_rate': random.uniform(*hyper_ranges['learning_rate']),
            'hidden_blocks': random.choice(hyper_ranges['hidden_blocks']),
            'layer_zero_depth': random.choice(hyper_ranges['layer_zero_depth']),
            'layer_scaler': random.choice(hyper_ranges['layer_scaler']),
            'dropout_rate': random.uniform(*hyper_ranges['dropout_rate']),
            'activation': random.choice(hyper_ranges['activation']),
            'normalization': random.choice(hyper_ranges['normalization'])
        }


        model, sub_dir = def_model(hyperparams=hyperparams, experimet_dir=experimet_dir, input_channel=input_channel, out_sz=output_sz, num_classes=num_classes)
        train_model(hyperparams=hyperparams, model=model, sub_dir=sub_dir, epochs=epochs)

if __name__ == "__main__":
    main()