from spade.behaviour import State
from spade.message import Message
import datetime
import random
import time
import uuid
import Config
import codecs
import pickle

class SendState(State):
    def pick_agent(self, agents: list[str]):
        agent = random.choice(agents)
        return agent
    
    async def send_message(self, recipient):
        id = str(uuid.uuid4())
        msg = Message(to=recipient)
        msg.set_metadata("conversation", "pre_consensus_data")
        #msg.set_metadata("message_id", id)

        local_weights = self.agent.weights
        local_losses = self.agent.losses

        if local_weights is not None or local_losses is not None:
            msg_local_weights = codecs.encode(pickle.dumps(local_weights), "base64").decode()
            msg_local_losses = codecs.encode(pickle.dumps(local_losses), "base64").decode()
            msg_max_order = str(round(self.agent.max_order, 3))

            content = msg_local_weights + "|" + msg_local_losses + "|" + msg_max_order
            
            msg.body = content
            msg.set_metadata("timestamp", str(datetime.datetime.now()))
            await self.send(msg)

    async def run(self):
        if len(self.agent.neighbours) > 0:
            agent = self.pick_agent(self.agent.neighbours)

            #print("Agent: " + self.agent.name + "Try to mail: " + agent)
            receiving_agent_name_domain = agent + Config.jid_domain
            await self.send_message(receiving_agent_name_domain)
            self.set_next_state("RECEIVE")
        else:
            self.set_next_state("TRAIN")