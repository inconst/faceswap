
import torch
import torchvision
import numpy as np
import random


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_grid_image(X):
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image_from_batches(Xs, Xt, Y, num_images):
    Xs = get_grid_image(Xs[:num_images])
    Xt = get_grid_image(Xt[:num_images])
    Y = get_grid_image(Y[:num_images])
    return torch.cat((Xs, Xt, Y), dim=1).numpy()