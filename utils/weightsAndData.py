
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle
import copy
import numpy as np

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


#Implementación de FLaMAS (Average)
def average_weights(w):
    """
    weights is a list of the dictionary of weight
    Returns the average of the weights.
    """
    w_avg = torch.load(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.load(w[i])[key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

#Implementación de CoL (Consensus)
def apply_consensus(own_weights, neighbour_weights, eps):
        """
        Apply the asynchronous consensus between the weights of the agent and those of its neighbour
        :param own_weights: weights of the agent
        :param neighbour_weights: weights of the neighbour
        :param eps: epsilon value, how much the neighbour weights accept the own
        :return: the new weights post-consensus
        """
        average_weights = copy.deepcopy(own_weights)
        for key in own_weights.keys():
            if len(own_weights[key]) != len(neighbour_weights[key]):
                print("Error - consensus can only be applied to arrays of same length")
                return None
            average_weights[key] = own_weights[key] + eps*(neighbour_weights[key] - own_weights[key])
        return average_weights



# weights = torch.load('model/mnist_cnn.pt')
# ##torch.save(model.state_dict(), "model/mnist_cnn.pt")
# weights = copy.deepcopy(weights)
# print(weights.keys())

# model1 = Net()
# model2 = Net()
# model2.load_state_dict(weights)
# #   Habria que retocar esto si en lugar de una red neuronal sencilla como un clasificador mnist,
# #       intentaramos entrenar un modelo mas grande.


# weights = [model2.state_dict(), model1.state_dict()]

# #w_avg = []
# w_avg = copy.deepcopy(weights[0])
# for key in w_avg.keys():
# for i in range(1, len(weights)):
#     print(weights[0][key], "+",weights[i][key], "=", w_avg[key] + weights[i][key])
#     w_avg[key] += weights[i][key]
# w_avg[key] = torch.div(w_avg[key], len(weights))

# model3 = Net()
# model3.load_state_dict(w_avg)