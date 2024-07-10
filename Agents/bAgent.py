#Basic Agent (MBAgent -- Model Bearing Agent)
import Config
import asyncio

from spade.agent import Agent
from spade.template import Template

from spade.behaviour import CyclicBehaviour, State, FSMBehaviour, OneShotBehaviour
from Agents.Behaviours import Training, Send, Receive, ReceiveBehaviour

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from Models.Training import *



class StateMachineBehaviour(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()

class MBAgent(Agent):
    def __init__(self, jid: str, password: str, verify_security: bool = False, model: torch.nn.Module = None, weights: dict = None,
            dataTrain:torchvision.datasets = None, dataTest:torchvision.datasets = None):
        super().__init__(jid, password, verify_security)
        
        self.trainer = FederatedLearning(model=model, weights=weights, dataTrain=dataTrain, dataTest=dataTest)
        self.neighbours = [x for x in Config.AGENT_NAMES if x != self.name]

        #self.available_agents = []

        self.weights = weights
        self.losses = None
        self.msg_max_order = 2

        #Maquina de estados
        self.state_machine_behaviour = None

        self.max_order = len(self.neighbours)

        self.trainer.build_Model()


    class Behav1(OneShotBehaviour):
        def on_available(self, jid, stanza):
            print("[{}] Agent {} is available.".format(self.agent.name, jid.split("@")[0]))

        def on_subscribed(self, jid):
            print("[{}] Agent {} has accepted the subscription.".format(self.agent.name, jid.split("@")[0]))
            print("[{}] Contacts List: {}".format(self.agent.name, self.agent.presence.get_contacts()))

        def on_subscribe(self, jid):
            print("[{}] Agent {} asked for subscription. Let's aprove it.".format(self.agent.name, jid.split("@")[0]))
            self.presence.approve(jid)
            self.presence.subscribe(jid)


        async def run(self):
            self.presence.on_subscribe = self.on_subscribe
            self.presence.on_subscribed = self.on_subscribed
            self.presence.on_available = self.on_available

            self.presence.set_available()
            for name in Config.AGENT_NAMES:
                if self.agent.name != name:
                    self.presence.subscribe(name + Config.jid_domain)

            print("ESTOY DISPONIBLE " + str(self.agent.presence.is_available()))
            print("Mis amigos " +  str(self.agent.presence.get_contacts()))


    async def setup(self):
        print("Agente inicializado")

        self.state_machine_behaviour = StateMachineBehaviour()

        self.state_machine_behaviour.add_state(name= "TRAIN", state= Training.TrainState(), initial=True)
        self.state_machine_behaviour.add_state(name= "SEND", state= Send.SendState())
        self.state_machine_behaviour.add_state(name= "RECEIVE", state= Receive.ReceiveState())
        
        self.state_machine_behaviour.add_transition(source= "TRAIN", dest= "SEND")

        self.state_machine_behaviour.add_transition(source= "SEND", dest= "RECEIVE")

        self.state_machine_behaviour.add_transition(source= "SEND", dest= "TRAIN")
        self.state_machine_behaviour.add_transition(source= "RECEIVE", dest= "TRAIN")

        
        state_machine_template = Template()
        state_machine_template.metadata = {"conversation": "pre_consensus_data"}

        self.add_behaviour(self.state_machine_behaviour, state_machine_template)

        #receive_template = Template()
        #receive_template.set_metadata("conversation", "pre_consensus_data")
        #self.receive_behaviour = ReceiveBehaviour2()
        #self.add_behaviour(self.receive_behaviour, receive_template)

        print("Agente finalizado el setup")

        self.add_behaviour(self.Behav1())