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


CONV_LAYERS = {
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "maxpool1d": nn.MaxPool1d,
    "maxpool2d": nn.MaxPool2d,
}

@dataclass
class CnnOps:
    layer: str
    dimension: int
    kernel_size: Union[int, Tuple[int]]
    in_channels: int
    out_channels: int
    input_size: Union[Tuple[int], List[int]] = None
    padding: Union[int, Tuple[int], List[int]] = 0
    stride: Union[int, Tuple[int], List[int]] = 1

    
    def __post_init__(self):
        print("layer in post init ", self.layer)
        assert self.layer in CONV_LAYERS.keys(), "layer must be one of {}".format(CONV_LAYERS.keys())
        if isinstance(self.kernel_size, int) and self.dimension == 2:
            self.kernel_size = [self.kernel_size, self.kernel_size]
        if isinstance(self.padding, int)  and self.dimension == 2:
            self.padding = [self.padding, self.padding]
        if isinstance(self.stride, int) and self.dimension == 2:
            self.stride = [self.stride, self.stride]
        self.set_out_shape()
        
    def set_out_shape(self):
        if "conv" in self.layer:
            self.conv_output()
        elif "pool" in self.layer:
            self.pool_output()
        print("out shape set to {}".format(self.out_shape))
            
    def pool_output(self):
        in_s = self.input_size
        pad, ks, st = self.padding, self.kernel_size, self.stride
        if self.dimension == 1:
            outshape = [int(np.floor((in_s + 2 * pad - 1 * (ks - 1) - 1) / st + 1))]
        elif self.dimension == 2:        
            h_out = int(np.floor((in_s[0] + 2 * pad[0] - 1 * (ks[0] - 1) - 1) / st[0] + 1))
            w_out = int(np.floor((in_s[1] + 2 * pad[1] - 1 * (ks[1] - 1) - 1) / st[1] + 1))
            outshape = [h_out, w_out]
        else:
            raise ValueError("Dimension must be 1 or 2")
        self.out_shape = [self.out_channels] + outshape

    def conv_output(self):
        in_s, pad, ks, st = self.input_size, self.padding, self.kernel_size, self.stride
        print(in_s, pad, ks, st)
        if self.dimension == 1:
            outshape = [np.floor((in_s + 2 * pad - (ks - 1) - 1) / st + 1).astype(int)]
        elif self.dimension == 2:
            outshape = [np.floor((in_s[0] + 2 * pad[0] - (ks[0] - 1) - 1) /
                                st[0] + 1).astype(int),
                        np.floor((in_s[1] + 2 * pad[1] - (ks[1] - 1) - 1) /
                                st[1] + 1).astype(int)]
        else:
            raise ValueError("Dimension must be 1 or 2")
        self.out_shape = [self.out_channels] + outshape
    


class Cnn(nn.Module):
    def __init__(self, input_size, output_size, cnn_architecture, mlp_architecture):
        super(Cnn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.cnn_architecture = cnn_architecture
        self.mlp_architecture = mlp_architecture
        
    def build(self):
        self.model, out_shape = self.build_cnn()
        mlp_input_shape = np.prod(out_shape)
        print("mlp_input_shape ", mlp_input_shape)
        self.classifier = Mlp(mlp_input_shape, self.output_size, self.mlp_architecture)
        self.classifier.build()
    
    def build_cnn(self):
        cnn_layers = []
        input_size = self.input_size
        
        for layer in self.cnn_architecture:
            layer["input_size"] = input_size
            l = CnnOps(**layer)
            if "conv" in l.layer:
                layer = CONV_LAYERS[l.layer](l.in_channels, l.out_channels, l.kernel_size,
                                             stride=l.stride, padding=l.padding)
            elif "pool" in l.layer:
                layer = CONV_LAYERS[l.layer](l.kernel_size, stride=l.stride,
                                             padding=l.padding)
            cnn_layers.append(layer)
            input_size = l.out_shape[-1]
        
        return nn.Sequential(*cnn_layers), l.out_shape
               
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)
    
if __name__ == "__main__":
    """Load and prepare data"""
    train_dataset = ExoDataset(add_channel=True)
    test_dataset = ExoDataset(is_train=False, resample=False, add_channel=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=False)
    """Create Model"""
    device = "cpu"
    cnn_architecture = [{"layer": "conv1d",
                         "dimension": 1,
                         "in_channels": 1,
                         "out_channels": 4,
                         "kernel_size": 3}, 
                        
                        {"layer": "maxpool1d",
                         "dimension": 1,
                         "in_channels": 4,
                         "out_channels": 4,
                         "kernel_size": 2},
                        
                        {"layer": "conv1d",
                         "dimension": 1,
                         "in_channels": 4,
                         "out_channels": 8,
                         "kernel_size": 3}
                        ]
    input_size = train_dataset[0][0].shape[-1]
    output_size = len(np.unique(train_dataset.y))
    mlp_architecture = [32, 64]
    model = Cnn(input_size, output_size, cnn_architecture, mlp_architecture)
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