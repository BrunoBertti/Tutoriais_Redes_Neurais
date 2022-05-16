########## CONJUNTOS DE DADOS E DATALOADERS ##########

    # O código para processar amostras de dados pode ficar confuso e difícil de manter; idealmente, queremos que nosso código de conjunto de dados seja desacoplado de nosso código de treinamento de modelo para melhor legibilidade e modularidade. O PyTorch fornece duas primitivas de dados: torch.utils.data.DataLoader e torch.utils.data.Dataset que permitem que você use conjuntos de dados pré-carregados, bem como seus próprios dados. O Dataset armazena as amostras e seus rótulos correspondentes, e o DataLoader envolve um iterável em torno do Dataset para facilitar o acesso às amostras.

    # As bibliotecas de domínio PyTorch fornecem vários conjuntos de dados pré-carregados (como FashionMNIST) que subclassificam torch.utils.data.Dataset e implementam funções específicas para os dados específicos. Eles podem ser usados para prototipar e comparar seu modelo. Você pode encontrá-los aqui: Conjuntos de dados de imagem, conjuntos de dados de texto e conjuntos de dados de áudio 


########## Carregando um conjunto de dados ##########


    # Aqui está um exemplo de como carregar o conjunto de dados Fashion-MNIST do TorchVision. Fashion-MNIST é um conjunto de dados de imagens de artigos de Zalando que consiste em 60.000 exemplos de treinamento e 10.000 exemplos de teste. Cada exemplo compreende uma imagem em tons de cinza 28×28 e um rótulo associado de uma das 10 classes.

    # Carregamos o conjunto de dados FashionMNIST com os seguintes parâmetros:
        
        # root é o caminho onde os dados de treinamento/teste são armazenados,
        
        # train especifica o conjunto de dados de treinamento ou teste,
        
        # download=True baixa os dados da internet se não estiver disponível na raiz.
        
        # transform e target_transform especificam as transformações de recursos e rótulos 



import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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