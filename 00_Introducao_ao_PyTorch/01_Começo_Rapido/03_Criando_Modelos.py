

########## COMEÇO RÁPIDO ##########

    # Esta seção é executada por meio da API para tarefas comuns em aprendizado de máquina. Consulte os links em cada seção para se aprofundar.



########## Trabalhando com dados ##########

    # O PyTorch tem duas primitivas para trabalhar com dados: torch.utils.data.DataLoader e torch.utils.data.Dataset. O Dataset armazena as amostras e seus rótulos correspondentes, e o DataLoader envolve um iterável em torno do Dataset. 

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

    # O PyTorch oferece bibliotecas específicas de domínio, como TorchText, TorchVision e TorchAudio, todas incluindo conjuntos de dados. Para este tutorial, usaremos um conjunto de dados TorchVision.

    # O módulo torchvision.datasets contém objetos Dataset para muitos dados de visão do mundo real como CIFAR, COCO (lista completa aqui). Neste tutorial, usamos o conjunto de dados FashionMNIST. Cada conjunto de dados do TorchVision inclui dois argumentos: transform e target_transform para modificar as amostras e os rótulos, respectivamente. 



# Baixe dados de treinamento de conjuntos de dados abertos. 
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Baixe dados de teste de conjuntos de dados abertos. 
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

    # Passamos o Dataset como argumento para o DataLoader. Isso envolve um iterável em nosso conjunto de dados e oferece suporte a lotes automáticos, amostragem, embaralhamento e carregamento de dados multiprocessos. Aqui definimos um tamanho de lote de 64, ou seja, cada elemento no iterável do carregador de dados retornará um lote de 64 recursos e rótulos. 

batch_size = 64


# Crie carregadores de dados. 

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Forma de X [N, C, H, W]: {X.shape}")
    print(f"Forma de y: {y.shape} {y.dtype}")
    break





########## Criando modelos ##########

    # Para definir uma rede neural no PyTorch, criamos uma classe que herda de nn.Module. Definimos as camadas da rede na função __init__ e especificamos como os dados passarão pela rede na função forward. Para acelerar as operações na rede neural, nós a movemos para a GPU, se disponível. 

# Obtenha um dispositivo cpu ou gpu para treinamento. 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando o dispositivo {device}")

# Definir modelo 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)