#Model Training
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset, DataLoader, Dataset

import pickle

#Contiene toda la información para poder 
class FederatedLearning:
    def __init__(self, model: torch.nn.Module = None, weights: dict = None, dataTrain:torchvision.datasets = None, dataTest:torchvision.datasets = None):
        self.model = model
        self.weight = weights

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None


        self.losses = None

        self.train_loader = DataLoader(dataTrain, batch_size=8, shuffle=True)
        self.test_loader = DataLoader(dataTest, batch_size=8, shuffle=False)

        # Check for CUDA (GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_Model(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def actualizeModel(self, stateDict):
        self.weight = stateDict
        self.model.load_state_dict(stateDict)

    async def train(self):
        # pickle.dumps para convertir a bits y poderse enviar
        epochs = 1
        self.model.train()
        t_losses = 0
        mini_batches_to_print = 100
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # print every n mini-batches:
                if mini_batches_to_print > 0 and i % mini_batches_to_print == mini_batches_to_print - 1:  
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / mini_batches_to_print}")
                    t_losses += running_loss
                    running_loss = 0.0
            t_losses += running_loss
        self.losses = t_losses
        self.weight = self.model.state_dict()

