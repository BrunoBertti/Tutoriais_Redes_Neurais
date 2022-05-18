

########## TRANSFORMAÇÕES ##########



    # Os dados nem sempre vêm em sua forma final processada, necessária para treinar algoritmos de aprendizado de máquina. Usamos transformações para realizar alguma manipulação dos dados e torná-los adequados para treinamento.

    # Todos os conjuntos de dados do TorchVision têm dois parâmetros -transform para modificar os recursos e target_transform para modificar os rótulos - que aceitam callables contendo a lógica de transformação. O módulo torchvision.transforms oferece várias transformações comumente usadas prontas para uso.

    # Os recursos do FashionMNIST estão no formato PIL Image e os rótulos são inteiros. Para treinamento, precisamos dos recursos como tensores normalizados e os rótulos como tensores codificados one-hot. Para fazer essas transformações, usamos ToTensor e Lambda. 


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)