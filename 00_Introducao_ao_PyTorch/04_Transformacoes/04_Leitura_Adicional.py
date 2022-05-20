

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


########## ToTensor() ##########


    # O ToTensor converte uma imagem PIL ou NumPy ndarray em um FloatTensor. e dimensiona os valores de intensidade de pixel da imagem no intervalo [0., 1.] 



########## Transformações Lambda ##########

    # As transformações lambda aplicam qualquer função lambda definida pelo usuário. Aqui, definimos uma função para transformar o inteiro em um tensor codificado one-hot. Ele primeiro cria um tensor zero de tamanho 10 (o número de rótulos em nosso conjunto de dados) e chama scatter_ que atribui um valor=1 no índice conforme fornecido pelo rótulo y. 

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))



 ########## Leitura adicional ##########   


    # https://pytorch.org/vision/stable/transforms.html