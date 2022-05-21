########## CONSTRUA A REDE NEURAL ##########


    # As redes neurais são compostas por camadas/módulos que realizam operações nos dados. O namespace torch.nn fornece todos os blocos de construção necessários para construir sua própria rede neural. Cada módulo no PyTorch subclasse o nn.Module. Uma rede neural é um módulo em si que consiste em outros módulos (camadas). Essa estrutura aninhada permite construir e gerenciar arquiteturas complexas facilmente.

    # Nas seções a seguir, construiremos uma rede neural para classificar imagens no conjunto de dados FashionMNIST. 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



########## Obter dispositivo para treinamento ##########


    # Queremos poder treinar nosso modelo em um acelerador de hardware como a GPU, se estiver disponível. Vamos verificar se o torch.cuda está disponível, senão continuamos a usar a CPU. 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



########## Defina a classe ##########

    # Definimos nossa rede neural subclassificando nn.Module e inicializamos as camadas da rede neural em __init__. Cada subclasse nn.Module implementa as operações nos dados de entrada no método forward. 


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


    # Criamos uma instância de NeuralNetwork e a movemos para o dispositivo e imprimimos sua estrutura. 


model = NeuralNetwork().to(device)
print(model)


    # Para usar o modelo, passamos os dados de entrada. Isso executa o encaminhamento do modelo, juntamente com algumas operações em segundo plano. Não chame model.forward() diretamente!

    # Chamar o modelo na entrada retorna um tensor de 10 dimensões com valores preditos brutos para cada classe. Obtemos as probabilidades de previsão passando por uma instância do módulo nn.Softmax. 


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")





########## Camadas de modelo ##########

    # Vamos dividir as camadas no modelo FashionMNIST. Para ilustrar, pegaremos um minilote de amostra de 3 imagens de tamanho 28x28 e veremos o que acontece com ele à medida que o passamos pela rede. 


input_image = torch.rand(3,28,28)
print(input_image.size())


########## nn.Flatten ##########

    # Inicializamos a camada nn.Flatten para converter cada imagem 2D 28x28 em uma matriz contígua de 784 valores de pixel (a dimensão do minilote (em dim=0) é mantida). 


flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())