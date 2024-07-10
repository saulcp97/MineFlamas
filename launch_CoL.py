# SPADE
# https://www.youtube.com/watch?v=jj5V-woERsg
import time
import os
import copy
from spade import wait_until_finished
from Models.Architectures import CIFAR8TinyCNN
from dataset.cifar import CIFAR8
import Config
import torchvision.transforms as transforms

import asyncio
import spade
from pathlib import Path

from Agents.bAgent import MBAgent



async def main():

    data = Path('data/cifar100_subset.pth').resolve()



    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    datasetTr = CIFAR8(root=data.parent.resolve(), train=True, transform=transform, download=True)
    datasetTe = CIFAR8(root=data.parent.resolve(), train=True, transform=transform, download=True)
    model = CIFAR8TinyCNN()
    weights = model.state_dict()

    names = Config.AGENT_NAMES
    pswrdd = "01234"
    agents = []
    for name in names:
        jid_name = name + Config.jid_domain
        agents.append(MBAgent(jid=jid_name, password=pswrdd,model=copy.deepcopy(model), weights=copy.deepcopy(weights), dataTrain=datasetTr, dataTest = datasetTe))
        

    for i in range(len(agents)):
        await agents[i].start()

    agents[0].web.start(hostname="127.0.0.1", port="10000")
    #Crear una lista de nombres y conexiones entre los agentes

if __name__ == "__main__":
    spade.run(main())