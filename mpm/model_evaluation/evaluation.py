import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def evaluate(model, data_loader, device, plot=False):
    model.eval()
    y_gt = []
    y_pr = []
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        res = model(x)
        y_pr.extend(torch.argmax(res, axis=1).detach().numpy())
        y_gt.extend(y.numpy())
    
    cm = confusion_matrix(y_gt, y_pr)
    if plot:
        plot_cm(cm, data_loader.dataset.mapping)
    return classification_report(y_gt,y_pr), cm

def plot_cm(cm, label_mapping):
    df_cm = pd.DataFrame(cm, index=list(label_mapping.keys()),
                          columns=list(label_mapping.keys()))
    sns.heatmap(df_cm, annot=True)   
    