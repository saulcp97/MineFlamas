#Test de la posibilidad de envio
import torch
from Models import Architectures
import pickle
import codecs

model = Architectures.ConvNet()

text = codecs.encode(pickle.dumps(model.state_dict()), "base64").decode()
print(text)

text2 = unpickled_neighbour_weights = pickle.loads(codecs.decode(text.encode(), "base64"))
print(text2)