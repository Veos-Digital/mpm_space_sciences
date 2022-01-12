import torch
from torch.optim import SGD, Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_weights(p0 = None, device="cpu", mean=0, std=2):
    if p0 is None:
        return torch.normal(mean=mean, std=std, size=(2,), dtype=torch.float64,
                            requires_grad=True, device=device)
    return torch.tensor(p0, dtype=torch.float64, requires_grad=True,
                        device=device)
    
def himmelblau(p):
    """Four identical local minima"""
    x, y = p
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


OPTIMIZERS_DICT = {
    "sgd": SGD,
    "adam": Adam,
    "rmsprop": RMSprop,
}

def get_optimizers(name, weights, lr):
    return OPTIMIZERS_DICT[name]([weights], lr=lr)


if __name__ == "__main__":
    n_iteration = 1000
    weights = get_weights()
    loss_surface = himmelblau
    optimizer_names = ["sgd", "adam", "rmsprop"]
    lr = 0.0001
    
    for optimizer_name in optimizer_names:
        optimizer = get_optimizers(optimizer_name, weights, lr)
        history = {'weights': [weights.cpu().detach().numpy()],
                   'loss': [loss_surface(weights.cpu().detach().numpy())]}


        for i in range(1, n_iteration + 1):
            loss = loss_surface(weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history['weights'].append(weights.cpu().detach().numpy())
            history['loss'].append(loss.cpu().detach().numpy())

            if (i % (n_iteration / 100) == 0 or i % n_iteration == 0):
                w = weights.cpu().detach().numpy()
                print("Iteration: {}\nWeights: {}\nLoss value: {}".format(i, w , loss_surface(weights)))

        print("Opt: {}\nLoss value: {} at {}".format(optimizer_name, loss_surface(weights), weights))