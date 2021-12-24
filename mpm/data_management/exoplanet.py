import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from mpm.constants import DATA_FOLDER
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def load_train_data(folder=None):
    folder = folder or DATA_FOLDER
    return pd.read_csv(os.path.join(folder, "exo", "exoTrain.csv"))
    
def load_test_data(folder=None):
    folder = folder or DATA_FOLDER
    return pd.read_csv(os.path.join(folder, "exo", "exoTest.csv"))

def plot_label_frequency(data, name=None, ax=None):
    sns.countplot(x='LABEL',data=data, ax=ax)
    if name is not None:
        plt.title(name)
    plt.xlabel('Planets')
    plt.ylabel('Frequency')


def separate_labels(data):
    X, y = data.drop(['LABEL'], axis=1), data['LABEL']
    return X, y


def plot_time_series(X, y, label=1, num_samples=5):
    X.loc[y==label][:num_samples].T.plot(subplots=True, figsize=(10,12))    


class ExoDataset(Dataset):
    def __init__(self, path_to_data=None, is_train=True, resample=True):
        self.path_to_data = path_to_data
        self.is_train = is_train
        if is_train:
            self.data = load_train_data()
        else:
            self.data = load_test_data()
            
        self.X, self.y = separate_labels(self.data)
        self.X, self.y = self.X.values, self.y.values
        self.X = self.X.astype(np.float32)
        if resample:
            self.X, self.y = self.smote_resampling(self.X, self.y)
        self.mapping = {v: i for i, v in enumerate(np.unique(self.y))}
        self.y = [self.mapping[v] for v in self.y]
    
    @staticmethod
    def smote_resampling(X, y):
        sm = SMOTE(random_state=42, sampling_strategy="minority")
        return sm.fit_resample(X, y)
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index]/self.X[index].max(), self.y[index]
        

if __name__ == "__main__":
    """Basic methods and summary stats
    """
    train_data, test_data = load_train_data(), load_test_data()
    f, axs = plt.subplots(1,2)
    plot_label_frequency(train_data, name="train", ax=axs[0])
    plot_label_frequency(test_data, name="test", ax=axs[1])
    x_train, y_train = separate_labels(train_data)
    plot_time_series(x_train, y_train)
    """Dataset and DataLoader"""
    train_dataset = ExoDataset(is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, 
                              shuffle=True, drop_last=False)