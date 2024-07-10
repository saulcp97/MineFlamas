import codecs
import datetime
import pickle
import Config
from spade.behaviour import CyclicBehaviour
from spade.message import Message


class ReceiveBehaviour(CyclicBehaviour):

    def consensus(self, msg):
        """
        Applies based on weights received from another agent
        :param msg: message containing the weights of the sender agent
        """
        if self.agent.weights is not None and msg.body.split("|")[0] != "None":
            t1 = datetime.datetime.now()
            self.agent.message_history.insert(0,
                                              "{}:{}:{} : Received message from {}".format(str(t1.hour), str(t1.minute),
                                                                            str(t1.second), str(msg.sender).split("/")[0]))
            if str(msg.sender).split("@")[0] in self.agent.message_statistics:
                self.agent.message_statistics[str(msg.sender).split("@")[0]]["receive"] += 1
            else:
                self.agent.message_statistics[str(msg.sender).split("@")[0]] = {"send": 0, "receive": 1}
            weights_and_losses = msg.body.split("|")
            neighbour_max_order = int(weights_and_losses[2])
            #print(f"[RECV-cyc] Consensus message: {weights_and_losses[0][:5]}...{weights_and_losses[0][-5:]} weights, {weights_and_losses[1][:5]}...{weights_and_losses[1][-5:]} losses, {neighbour_max_order} max order")
            unpickled_neighbour_weights = pickle.loads(codecs.decode(weights_and_losses[0].encode(), "base64"))
            unpickled_neighbour_losses = pickle.loads(codecs.decode(weights_and_losses[1].encode(), "base64"))
            # print(neighbour_max_order)
            if self.agent.max_order < neighbour_max_order:
                self.agent.max_order = neighbour_max_order
                self.agent.epsilon_logger.write_to_file(str(self.agent.max_order))

            unpickled_local_weights = pickle.loads(codecs.decode(self.agent.weights.encode(), "base64"))
            # print(unpickled_neighbour_weights[0]['layer_input.weight'])

            # Apply consensus and update model
            consensus_weights = self.agent.consensus.apply_consensus(unpickled_local_weights,
                                                                     unpickled_neighbour_weights,
                                                                     1 / self.agent.max_order)

            #self.agent.weight_logger.write_to_file(
            #    "CONSENSUS,{},{},{},{}".format(consensus_weights[0]['layer_input.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_input.bias'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.weight'].numpy().flatten()[0],
            #                                   consensus_weights[0]['layer_hidden.bias'].numpy().flatten()[0]))

            self.agent.federated_learning.add_new_local_weight_local_losses(consensus_weights[0],
                                                                            unpickled_neighbour_losses)
            self.agent.federated_learning.set_model()

            # Update agent properties
            self.agent.weights = codecs.encode(pickle.dumps(consensus_weights), "base64").decode()
            self.agent.losses = codecs.encode(pickle.dumps(unpickled_neighbour_losses), "base64").decode()

    async def send_message(self, recipient, response_id):
        msg = Message(to=str(recipient))  # Instantiate the message
        msg.set_metadata("conversation", "response_data")
        msg.set_metadata("message_id", response_id)

        if recipient in self.agent.message_statistics:
            self.agent.message_statistics[recipient]["send"] += 1
        else:
            self.agent.message_statistics[recipient] = {"send": 1, "receive": 0}
        self.agent.message_logger.write_to_file("SEND,{},{}".format(response_id, recipient))

        local_weights = self.agent.weights
        local_losses = self.agent.losses

        if local_weights is None or local_losses is None:
            msg.body = "I don't have any weights yet"
            print("[{}] Sending message to {}".format(self.agent.name, recipient))
            msg.set_metadata("timestamp", str(datetime.datetime.now()))
            await self.send(msg)
        else:
            msg_local_weights = str(local_weights).strip()
            msg_local_losses = str(local_losses).strip()
            msg_max_order = str(round(self.agent.max_order, 3))
            content = msg_local_weights + "|" + msg_local_losses + "|" + msg_max_order
            print("[{}] Sending message to {}".format(self.agent.name, recipient))
            print(f"[SEND-cyc] Message length: {len(msg_local_weights)} weights + {len(msg_local_losses)} losses + {len(msg_max_order)} max order = {len(content)}")
            print(f"[SEND-cyc] Message content: {msg_local_weights[:5]}...{msg_local_weights[-5:]} weights, {msg_local_losses[:5]}...{msg_local_losses[-5:]} losses, {msg_max_order} max order")
                
            multipart_messages = self.multipart_handler.generate_multipart_messages(content, Config.max_message_body_length, msg)            
            if multipart_messages is not None:
                for i, message in enumerate(multipart_messages):
                    print(f"[SEND-cyc]  multipart message ({i + 1}/{len(multipart_messages)}) sent to {recipient}")
                    message.set_metadata("timestamp", str(datetime.datetime.now()))
                    await self.send(message)
            else:
                msg.body = content
                msg.set_metadata("timestamp", str(datetime.datetime.now()))
                await self.send(msg)


    async def run(self):
        """
        Waits until a message is received, then calls a method to apply consensus based on the weights contained
        in the message.
        """
        # Wait for a message
        msg = await self.receive(timeout=4)
        if msg:
            self.agent.message_logger.write_to_file("RECEIVE,{},{}".format(msg.get_metadata("message_id"), msg.sender))
            now = datetime.datetime.now()
            msg_timestamp = msg.get_metadata("timestamp")
            difference = now - datetime.datetime.strptime(msg_timestamp, "%Y-%m-%d %H:%M:%S.%f")
            difference_seconds = difference.total_seconds()

            if difference_seconds < Config.max_seconds_pre_consensus_message:
                # We need to keep in memory the last message received by the agent
                sender = str(msg.sender)
                self.agent.last_message = msg

                multipart = self.multipart_handler.rebuild_multipart(msg)
                if multipart is not None:
                    msg = multipart

                # response_msg = Message(to=sender)
                # response_msg.body = str(self.agent.weights).strip() + "|" + str(self.agent.losses).strip() + "|" + str(
                #     round(self.agent.max_order, 3))
                # response_msg.set_metadata("conversation", "response_data")
                # response_msg.set_metadata("timestamp", str(datetime.datetime.now()))
                # response_msg.set_metadata("message_id", self.agent.last_message.get_metadata("message_id"))

                # We apply the consensus if the agent is not training
                if self.agent.state_machine_behaviour.current_state != Config.TRAIN_STATE_AG:
                    if not self.multipart_handler.is_multipart(msg) or multipart is not None:
                        self.consensus(msg)
                else:
                    if not self.multipart_handler.is_multipart(msg) or multipart is not None:
                        self.agent.pending_consensus_messages.append(msg)

                if not self.multipart_handler.is_multipart(msg) or multipart is not None:
                    t1 = datetime.datetime.now()
                    self.agent.message_history.insert(0, "{}:{}:{} : Sent response to {}".format(str(t1.hour),
                                                                                             str(t1.minute),
                                                                                             str(t1.second),
                                                                                             str(self.agent.last_message.sender).split("/")[0]))
                    await self.send_message(sender, msg.get_metadata("message_id"))
                # await self.send(response_msg)
            else:
                print("[{}] Received old message".format(self.agent.name))
        # print("New iteration")
        # await asyncio.sleep(0.1)