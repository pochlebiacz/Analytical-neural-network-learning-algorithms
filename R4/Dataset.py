import torch
import torch.nn as nn

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# V = activation(model.fc1.bias.data.reshape(-1, 1) + model.fc1.weight.data @ interset[:][0].reshape(1, -1))
# H = torch.concatenate([torch.ones(1, len(interset)), V], dim=0).T
# Y = interset[:][1].reshape(1, -1).T
# W2 = torch.linalg.pinv(H) @ Y.reshape(1, -1, 1)
# model.fc2.bias.data, model.fc2.weight.data = W2[0][0], W2[0][1:].T