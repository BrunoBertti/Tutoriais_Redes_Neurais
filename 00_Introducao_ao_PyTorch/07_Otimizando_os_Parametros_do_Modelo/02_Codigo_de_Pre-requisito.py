
########## OTIMIZAÇÃO DOS PARÂMETROS DO MODELO ##########



    # Agora que temos um modelo e dados, é hora de treinar, validar e testar nosso modelo otimizando seus parâmetros em nossos dados. O treinamento de um modelo é um processo iterativo; em cada iteração (chamada de época) o modelo faz uma estimativa sobre a saída, calcula o erro em sua estimativa (perda), coleta as derivadas do erro em relação aos seus parâmetros (como vimos na seção anterior) e otimiza esses parâmetros usando gradiente descendente. Para um passo a passo mais detalhado desse processo, confira este vídeo sobre retropropagação de 3Blue1Brown. 


########## Código de pré-requisito ##########


    # Carregamos o código das seções anteriores em Datasets & DataLoaders e Build Model. 


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()