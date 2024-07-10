#Comportamiento de entrenamiento

from spade.behaviour import State
import time

class TrainState(State):

    async def train_local(self):
        self.start_time = time.monotonic()
        await self.agent.trainer.train()
        self.agent.losses = self.agent.trainer.losses
        self.agent.weights = self.agent.trainer.weight
        self.end_time = time.monotonic()


    async def run(self):
        await self.train_local()

        self.set_next_state("SEND")
