#Receive
from spade.behaviour import State
from spade.message import Message

import pickle
import codecs
from utils.weightsAndData import apply_consensus
class ReceiveState(State):
    
    async def consensus(self, msg):
        if self.agent.weights is not None and msg.body.split("|")[0] != "None" and not msg.body.startswith("I don't"):
            # Process message
            weights_and_losses = msg.body.split("|")
            neighbour_max_order = int(weights_and_losses[2])

            #print(f"[RECV-fsm] Consensus message: {weights_and_losses[0][:5]}...{weights_and_losses[0][-5:]} weights, {weights_and_losses[1][:5]}...{weights_and_losses[1][-5:]} losses, {neighbour_max_order} max order")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            # print(neighbour_max_order)
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))

            # Apply consensus and update model
            consensus_weights = apply_consensus(unpickled_local_weights, unpickled_neighbour_weights, 1 / self.agent.max_order)

            #self.agent.weight_logger.write_to_file(
            #    "CONSENSUS,{},{},{},{}".format(consensus_weights[0]['layer_input.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_input.bias'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.bias'].numpy().flatten()[0]))

            # Update agent properties
            self.agent.weights = codecs.encode(pickle.dumps(consensus_weights), "base64").decode()
            self.agent.trainer.actualizeModel(self.agent.weights)
            self.agent.losses = codecs.encode(pickle.dumps(unpickled_neighbour_losses), "base64").decode()


    async def run(self):
        msg = await self.receive(timeout=20) #Segundos

        if not msg:
            self.set_next_state("TRAIN")
        else:
            self.consensus(msg)
            self.set_next_state("TRAIN")