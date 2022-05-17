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



########## Iterando e Visualizando o Conjunto de Dados ##########


    # Podemos indexar conjuntos de dados manualmente como uma lista: training_data[index]. Usamos matplotlib para visualizar algumas amostras em nossos dados de treinamento. 


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()





########## Criando um conjunto de dados personalizado para seus arquivos ##########


    # Uma classe de conjunto de dados personalizada deve implementar três funções: __init__, __len__ e __getitem__. Dê uma olhada nesta implementação; as imagens FashionMNIST são armazenadas em um diretório img_dir, e seus rótulos são armazenados separadamente em um arquivo CSV annotations_file.

    # Nas próximas seções, detalharemos o que está acontecendo em cada uma dessas funções. 


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label