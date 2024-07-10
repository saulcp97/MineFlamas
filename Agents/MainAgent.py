#Server Agent in FLaMAS
import Config
import asyncio

from spade.agent import Agent
from spade.message import Message

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from Models.Training import *
from spade.behaviour import CyclicBehaviour, State, FSMBehaviour, OneShotBehaviour


class PresenceBehav(OneShotBehaviour):

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

class send_state(State):
    def send_message(self, recipient):
        msg = Message(to=recipient)  # Instantiate the message

        local_weights0 = self.get('local_weights')
        local_losses0 = self.get('local_losses')

        if local_weights0 == None and local_losses0 == None:
            msg.body = 'CoLearning: ' + 'No Data'
            msg.set_metadata('conversation', 'CoLearning')
            return msg
        else:
            print("Send Message to ", recipient)
            # https://stackoverflow.com/questions/30469575/how-to-pickle-and-unpickle-to-portable-string-in-python-3
            # msg.body = '{' + '"' + "local_weights:" + '"' + local_weights0 + '"' + "," + "value1" + ":" + '"' + local_losses0 + '"' + '}'
            msg.body = '' + str(local_weights0).strip()
            msg.set_metadata('conversation', 'CoLearning')
            return msg

    async def run(self):
        print("-    This is the sender state")
        contact_list = self.agent.presence.get_contacts()
        # print(type(contact_list))
        # print(list(contact_list.values()))

        for i_contact in list(contact_list):
            myContactList = i_contact[0]
            recipient = myContactList + "@" + Config.xmpp_server
            msg = self.send_message(recipient)
            await self.send(msg)

class StateMachineBehaviour(FSMBehaviour):
    async def on_start(self):
        print(f"FSM starting at initial state {self.current_state}")

    async def on_end(self):
        print(f"FSM finished at state {self.current_state}")
        await self.agent.stop()

class ServerAgent(Agent):
    async def setup(self):
        fsm = StateMachineBehaviour()


        self.add_behaviour(fsm)

        self.add_behaviour(PresenceBehav())
