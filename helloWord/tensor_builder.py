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
from tqdm import tqdm
import os 
import medication_data_maker


class CombinedDataSet(Dataset):
    def __init__(self, raw_data_path):
        X, y = torch.load(raw_data_path, weights_only=True))
        X = X.unsqueeze(1)
        self.x = X
        self.y = y
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# def create_combined_dataset():
#     raw_data_dir = '/home/kuba/Documents/Data/Raw/RatEEG_R/' #THIS NEEDS TO BE ABSOLUTE PATH

#     all_datasets = []
#     for file_name in tqdm(sorted(os.listdir(raw_data_dir))):
#         if file_name.endswith('.pt'):
#             raw_data_path = os.path.join(raw_data_dir, file_name)
#             dataset = CombinedDataSet(raw_data_path)
#             all_datasets.append(dataset)

#     return ConcatDataset(all_datasets)

def getDataSet(randomState=69, trainPercent=0.7, window_size=300, stride=50, flatten=True):
    combined_dataset = medication_data_maker.create_combined_dataset()

    # split into train and test
    total_size = len(combined_dataset)
    train_size = int(trainPercent * total_size)
    test_size = total_size - train_size

    generator = torch.Generator().manual_seed(randomState)
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size], generator=generator)

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, test_dataset = getDataSet()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
