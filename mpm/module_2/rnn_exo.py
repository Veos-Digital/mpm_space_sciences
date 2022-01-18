import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Union, Tuple, List
from mpm.data_management.exoplanet import ExoDataset
from mpm.model_evaluation.evaluation import evaluate
from mpm.module_1.mlp_exo import Mlp, train
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


RNN_LAYERS = {
    "lstm": nn.LSTM,
    "gru": nn.GRU,
}


class RnnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 mlp_architecture, layer_type="lstm"):
        super(RnnClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.mlp_architecture = mlp_architecture
        
    def build(self):
        self.rnn = RNN_LAYERS[self.layer_type](self.input_size, self.hidden_size, self.num_layers,
                                               batch_first=True)
        self.classifier = Mlp(self.hidden_size, self.output_size,
                              self.mlp_architecture)
        self.classifier.build()
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.classifier(out[:, -1, :])
        return out
    
if __name__ == "__main__":
    
    """Load and prepare data"""
    train_dataset = ExoDataset(add_channel=True)
    test_dataset = ExoDataset(is_train=False, resample=False, add_channel=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=False)
    """Create Model"""
    device = "cpu"
    input_size = train_dataset[0][0].shape[-1]
    output_size = len(np.unique(train_dataset.y))
    hidden_size = 64
    num_layers = 1
    mlp_architecture = [64, 64]
    model = RnnClassifier(input_size, hidden_size, num_layers, output_size,
                          mlp_architecture, layer_type="lstm")
    model.build()
    """Loss function"""
    loss_func = nn.CrossEntropyLoss(reduction="mean")
    """Optimizer and parameters"""
    lr = 0.01
    optimizer = torch.optim.Adam
    """Train"""
    model = train(model, train_loader, loss_func, optimizer, lr, num_epochs=10,
                  device="cpu", test_loader=test_loader)
    """Evaluate"""
    train_report, train_cm = evaluate(model, train_loader, device, plot=False)
    test_report, test_cm = evaluate(model, test_loader, device, plot=True)
    print("Train report:\n{}".format(train_report))
    print("Test report:\n{}".format(test_report))
    plt.show()