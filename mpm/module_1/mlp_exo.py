import numpy as np
import torch
import torch.nn as nn
from mpm.data_management.exoplanet import ExoDataset
from mpm.model_evaluation.evaluation import evaluate, plot_cm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Mlp(nn.Module):
    def __init__(self, input_size, output_size, architecture):
        super(Mlp, self).__init__()
        self.layer_dims = [input_size] + architecture + [output_size]
        
    def build(self):
        layers = [nn.Linear(in_units, out_units)
                  for in_units, out_units in zip(self.layer_dims, self.layer_dims[1:])]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train(model, train_loader, loss_func, optimizer, lr, num_epochs=10, device="cpu",
          test_loader=None):
    optimizer = optimizer(model.parameters(), lr= lr)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = []
        correct = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            res = model(x)
            correct += torch.sum(torch.argmax(res, axis=1) == y)
            loss = loss_func(res, y)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            
        print('Epoch {}\nloss: {:.3f}\taccuracy: {}'.format(epoch + 1, 
              np.mean(running_loss),
              100 * correct/len(train_loader.dataset.y)))
        
        if test_loader is not None:
            test(model, test_loader, loss_func, device="cpu")
        
    return model


def test(model, test_loader, loss_func, device="cpu"):
    test_running_loss, test_correct = [], 0
    model.eval()
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        res = model(x)
        test_correct += torch.sum(torch.argmax(res, axis=1) == y)
        loss = loss_func(res, y)
        test_running_loss.append(loss.item())
    
    print('Test\nloss: {:.3f}\taccuracy: {}'.format( 
        np.mean(test_running_loss),
        100 * test_correct/len(test_loader.dataset.y)))

if __name__ == "__main__":
    """Load and prepare data"""
    train_dataset, test_dataset = ExoDataset(), ExoDataset(is_train=False, resample=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, drop_last=False)
    """Create Model"""
    device = "cpu"
    input_size = train_dataset[0][0].shape[0]
    output_size = len(np.unique(train_dataset.y))
    architecture = [128, 256, 512]
    model = Mlp(input_size, output_size, architecture)
    model.build()
    model = model.to(device)
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