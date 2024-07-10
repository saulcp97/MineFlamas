#Communications Test


import getpass
import asyncio

import spade
from spade.template import Template
from spade.message import Message
import datetime

template = Template()
template.metadata = {"conversation": "pre_consensus_data"}

message = Message()
message.sender = "sender1@host"
message.to = "recv1@host"
message.body = "Hello World"
message.set_metadata("conversation", "pre_consensus_data")
message.set_metadata("timestamp", str(datetime.datetime.now()))
assert template.match(message)

