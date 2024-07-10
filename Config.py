# Configuración para el sistema.


xmpp_server = "gtirouter.dsic.upv.es"




web_port = 3000
url = "localhost"
jid_domain = "@" + xmpp_server

# FSM name states Central Federado
SETUP_STATE_CFDL = "SETUP_STATE"
STOP_STATE_CFDL = "STOP_STATE"
SEND_STATE_CFDL = "SEND_STATE"

# Net Configuration Path File
path_csv = 'Network_Structures/Connection_1.csv'

# Data-Set Path
data_set_path = "../data"

CONSENSUS_LOGGER = "CONSENSUS_LOGGER"
MESSAGE_LOGGER = "MESSAGE_LOGGER"


#   Network Structures (8 miembros)
# 4 coalitions 1~3 members each coalition
COALITION_PROBABILITY = 0.75

AGENT_NAMES = ["scp_0", "scp_1", "scp_2", "scp_3", "scp_4", "scp_5"]

#AGENT_NAMES = ["a0", "a1"]