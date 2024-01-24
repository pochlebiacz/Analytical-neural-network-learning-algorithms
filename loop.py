import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def loop(trainset, testset):
    activation = nn.Sigmoid()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    max_train_error = []
    min_train_error = []
    max_test_error = []
    min_test_error = []
    num_neurons = 200
    for j in range(1, num_neurons+1):
        print(j)
        train_error = []
        test_error = []
        for i in range(10):
            model = ELM(2, 5*j, 1).to(device)
            V = activation(model.fc1.bias.data.reshape(-1, 1) + model.fc1.weight.data @ trainset[:][0].T.to(device))
            H = torch.concatenate([torch.ones(1, len(trainset)).to(device), V.to(device)], dim=0).T
            T = trainset[:][1].reshape(1, -1).T.to(device)
            # ---------------- bez regularyzacji ----------------
            W2 = torch.linalg.pinv(H) @ T.reshape(1, -1, 1) 
            # ---------------- z regularyzacją ------------------
            # P = torch.linalg.inv(H.T @ H + 1e-3*torch.eye(H.shape[1]).to(device))
            # W2 = P @ H.T @ T.reshape(1, -1, 1)
            # ---------------------------------------------------
            model.fc2.bias.data, model.fc2.weight.data = W2[0][0], W2[0][1:].T
            train_error.append((model(trainset[:][0].clone().detach().reshape(-1, 2).to(device))-trainset[:][1].reshape(-1, 1).to(device)).reshape(-1).pow(2).mean())
            test_error.append((model(testset[:][0].clone().detach().reshape(-1, 2).to(device))-testset[:][1].reshape(-1, 1).to(device)).reshape(-1).pow(2).mean())
        max_train_error.append(max(train_error).item())
        min_train_error.append(min(train_error).item())
        max_test_error.append(max(test_error).item())
        min_test_error.append(min(test_error).item())
    ax[0].plot([5*i+5 for i in range(num_neurons)], max_train_error, 'r', label='Błąd maksymalny')
    ax[0].plot([5*i+5 for i in range(num_neurons)], min_train_error, 'b', label='Błąd minimalny')
    ax[0].set_ylabel('Błąd średniokwadratowy zbioru uczącego')
    ax[0].set_xlabel('Liczba neuronów')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend(loc='upper right')
    ax[1].plot([5*i+5 for i in range(num_neurons)], max_test_error, 'r', label='Błąd maksymalny')
    ax[1].plot([5*i+5 for i in range(num_neurons)], min_test_error, 'b', label='Błąd minimalny')
    ax[1].set_ylabel('Błąd średniokwadratowy zbioru weryfikującego')
    ax[1].set_xlabel('Liczba neuronów')
    ax[1].set_yscale('log')
    ax[1].grid()
    ax[1].legend(loc='upper right')
    plt.show()