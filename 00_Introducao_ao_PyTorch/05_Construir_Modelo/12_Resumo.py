##### Tutorial retirado de: https://pytorch.org/tutorials/beginner/basics/intro.html


########## APRENDA O BÁSICO ##########

    # Autores: Suraj Subramanian, Seth Juarez, Cassie Breviu, Dmitry Soshnikov, Ari Bornstein

    # A maioria dos fluxos de trabalho de aprendizado de máquina envolve trabalhar com dados, criar modelos, otimizar parâmetros de modelo e salvar os modelos treinados. Este tutorial apresenta um fluxo de trabalho de ML completo implementado no PyTorch, com links para saber mais sobre cada um desses conceitos.

    # Usaremos o conjunto de dados FashionMNIST para treinar uma rede neural que prevê se uma imagem de entrada pertence a uma das seguintes classes: camiseta/top, calça, pulôver, vestido, casaco, sandália, camisa, tênis, bolsa ou tornozelo Bota.

    # Este tutorial pressupõe uma familiaridade básica com os conceitos de Python e Deep Learning. 



########## Executando o código do tutorial ##########


    # Você pode executar este tutorial de duas maneiras:

        # Na nuvem: esta é a maneira mais fácil de começar! Cada seção tem um link “Executar no Microsoft Learn” na parte superior, que abre um bloco de anotações integrado no Microsoft Learn com o código em um ambiente totalmente hospedado.
        
        # Localmente: Esta opção requer que você configure o PyTorch e o TorchVision primeiro em sua máquina local (instruções de instalação). Baixe o notebook ou copie o código em seu IDE favorito. 




########## Como usar este guia ##########

    # Se você estiver familiarizado com outras estruturas de aprendizado profundo, confira primeiro o 0. Quickstart para se familiarizar rapidamente com a API do PyTorch.


    # Se você é novo em estruturas de aprendizado profundo, vá direto para a primeira seção do nosso guia passo a passo: 1. Tensores.

    # 0. Início rápido
    # 1. Tensores
    # 2. Conjuntos de dados e DataLoaders
    # 3. Transformações
    # 4. Construir Modelo
    # 5. Diferenciação Automática
    # 6. Ciclo de Otimização
    # 7. Salvar, carregar e usar o modelo 





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




########## Otimizando os parâmetros do modelo ##########

    # Para treinar um modelo, precisamos de uma função de perda e de um otimizador. 


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    # Em um único loop de treinamento, o modelo faz previsões no conjunto de dados de treinamento (alimentado a ele em lotes) e retropropaga o erro de previsão para ajustar os parâmetros do modelo. 

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Calcular erro de previsão 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    # Também verificamos o desempenho do modelo em relação ao conjunto de dados de teste para garantir que ele esteja aprendendo. 

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Erro de teste: \n Precisão: {(100*correct):>0.1f}%, Perda média: {test_loss:>8f} \n")



    # O processo de treinamento é realizado em várias iterações (épocas). Durante cada época, o modelo aprende parâmetros para fazer melhores previsões. Imprimimos a precisão e perda do modelo em cada época; gostaríamos de ver a precisão aumentar e a perda diminuir a cada época. 


epochs = 5
for t in range(epochs):
    print(f"iterações {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Feito!")






########## Salvando modelos ##########


    # Uma maneira comum de salvar um modelo é serializar o dicionário de estado interno (contendo os parâmetros do modelo). 


torch.save(model.state_dict(), "model.pth")
print("Modelo salvo.pth")




########## Carregando modelos ##########

    # O processo para carregar um modelo inclui recriar a estrutura do modelo e carregar o dicionário de estado nela. 


model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


    # Este modelo pode agora ser usado para fazer previsões. 


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Previsto: "{predicted}", Real: "{actual}"')



########## TENSORES ##########

    # Tensores são uma estrutura de dados especializada que são muito semelhantes a arrays e matrizes. No PyTorch, usamos tensores para codificar as entradas e saídas de um modelo, bem como os parâmetros do modelo.

    # Os tensores são semelhantes aos ndarrays do NumPy, exceto que os tensores podem ser executados em GPUs ou outros aceleradores de hardware. Na verdade, tensores e matrizes NumPy podem frequentemente compartilhar a mesma memória subjacente, eliminando a necessidade de copiar dados (consulte Ponte com NumPy). Os tensores também são otimizados para diferenciação automática (veremos mais sobre isso posteriormente na seção Autograd). Se você estiver familiarizado com ndarrays, estará em casa com a API do Tensor. Se não, acompanhe! 


import torch
import numpy as np



########## Inicializando um tensor ##########

    # Os tensores podem ser inicializados de várias maneiras. Dê uma olhada nos exemplos a seguir:



##### Diretamente dos dados

    # Os tensores podem ser criados diretamente dos dados. O tipo de dados é inferido automaticamente. 

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


##### De uma matriz NumPy

    # Tensores podem ser criados a partir de matrizes NumPy (e vice-versa - veja Ponte com NumPy). 

np_array = np.array(data)
x_np = torch.from_numpy(np_array)


##### De outro tensor:

    # O novo tensor retém as propriedades (forma, tipo de dados) do tensor de argumento, a menos que seja explicitamente substituído. 

x_ones = torch.ones_like(x_data) # retém as propriedades de x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # substitui o tipo de dados de x_data 
print(f"Random Tensor: \n {x_rand} \n")


##### Com valores aleatórios ou constantes:

    # forma é uma tupla de dimensões de tensor. Nas funções abaixo, determina a dimensionalidade do tensor de saída. 

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")




########## Atributos de um tensor ##########

    # Os atributos do tensor descrevem sua forma, tipo de dados e o dispositivo no qual estão armazenados. 

tensor = torch.rand(3,4)

print(f"Forma do tensor: {tensor.shape}")
print(f"Tipo de dados do tensor: {tensor.dtype}")
print(f"O tensor do dispositivo é armazenado em: {tensor.device}")





########## Operações em tensores ##########


    # Mais de 100 operações de tensor, incluindo aritmética, álgebra linear, manipulação de matrizes (transposição, indexação, fatiamento), amostragem e muito mais são descritas de forma abrangente aqui.

    # Cada uma dessas operações pode ser executada na GPU (normalmente em velocidades mais altas do que em uma CPU). Se você estiver usando o Colab, aloque uma GPU acessando Runtime > Change runtime type > GPU.

    # Por padrão, os tensores são criados na CPU. Precisamos mover explicitamente os tensores para a GPU usando o método .to (após verificar a disponibilidade da GPU). Tenha em mente que copiar tensores grandes entre dispositivos pode ser caro em termos de tempo e memória! 


# Movemos nosso tensor para a GPU, se disponível
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

    # Experimente algumas das operações da lista. Se você estiver familiarizado com a API NumPy, achará a API do Tensor muito fácil de usar. 


##### Indexação e fatiamento padrão do tipo numpy: 

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)


##### Unindo tensores Você pode usar torch.cat para concatenar uma sequência de tensores ao longo de uma determinada dimensão. Veja também torch.stack, outro tensor unindo op que é sutilmente diferente de torch.cat. 

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


##### Operaçoes aritimeticas 


# Isso calcula a multiplicação da matriz entre dois tensores. y1, y2, y3 terão o mesmo valor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# Isso calcula o produto por elemento. z1, z2, z3 terão o mesmo valor 
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)


##### Tensores de elemento único Se você tiver um tensor de um elemento, por exemplo, agregando todos os valores de um tensor em um valor, você pode convertê-lo em um valor numérico Python usando item(): 

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))



##### Operações in-loco As operações que armazenam o resultado no operando são chamadas in-loco. Eles são indicados por um sufixo _. Por exemplo: x.copy_(y), x.t_(), mudará x. 

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)


    # NOTA: As operações in-loco economizam alguma memória, mas podem ser problemáticas ao calcular derivativos devido a uma perda imediata do histórico. Por isso, seu uso é desencorajado. 





########## Ponte com NumPy ##########

    # Tensores nas matrizes CPU e NumPy podem compartilhar seus locais de memória subjacentes e alterar um alterará o outro. 






########## Tensor para matriz NumPy ##########

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


    # Uma mudança no tensor reflete na matriz NumPy. 

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")




########## Matriz NumPy para tensor  ##########


n = np.ones(5)
t = torch.from_numpy(n)

    # As alterações na matriz NumPy refletem no tensor. 

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


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



########## __init__ ##########

    # A função __init__ é executada uma vez ao instanciar o objeto Dataset. Inicializamos o diretório que contém as imagens, o arquivo de anotações e ambas as transformações (abordadas com mais detalhes na próxima seção).

    # O arquivo labels.csv se parece com: 

# tshirt1.jpg, 0
# tshirt2.jpg, 0
# ......
# ankleboot999.jpg, 9


def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform


########## __len__ ##########


    # A função __len__ retorna o número de amostras em nosso conjunto de dados.

    # Exemplo: 


def __len__(self):
    return len(self.img_labels)



########## __getitem__ ##########


    # A função __getitem__ carrega e retorna uma amostra do conjunto de dados no índice idx fornecido. Com base no índice, ele identifica a localização da imagem no disco, converte isso em um tensor usando read_image, recupera o rótulo correspondente dos dados csv em self.img_labels, chama as funções de transformação neles (se aplicável) e retorna a imagem do tensor e rótulo correspondente em uma tupla. 

def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label


########## Preparando seus dados para treinamento com DataLoaders ##########


    # O conjunto de dados recupera os recursos do nosso conjunto de dados e rotula uma amostra por vez. Ao treinar um modelo, normalmente queremos passar amostras em “minilotes”, reorganizar os dados a cada época para reduzir o overfitting do modelo e usar o multiprocessamento do Python para acelerar a recuperação de dados.

    # DataLoader é um iterável que abstrai essa complexidade para nós em uma API fácil. 


from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)



########## Iterar através do DataLoader ##########


    # Carregamos esse conjunto de dados no DataLoader e podemos iterar pelo conjunto de dados conforme necessário. Cada iteração abaixo retorna um lote de train_features e train_labels (contendo batch_size=64 recursos e rótulos, respectivamente). Como especificamos shuffle=True, depois de iterarmos em todos os lotes, os dados são embaralhados (para um controle mais refinado sobre a ordem de carregamento de dados, dê uma olhada em Samplers). 

# Exibe a imagem e o rótulo. 
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


########## Iterar através do DataLoader ##########

    # https://pytorch.org/docs/stable/data.html




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


########## nn.Linear ##########

    # A camada linear é um módulo que aplica uma transformação linear na entrada usando seus pesos e vieses armazenados. 

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


########## nn.ReLU ##########

    # As ativações não lineares são o que criam os mapeamentos complexos entre as entradas e saídas do modelo. Eles são aplicados após transformações lineares para introduzir não linearidade, ajudando as redes neurais a aprender uma ampla variedade de fenômenos.

    # Neste modelo, usamos nn.ReLU entre nossas camadas lineares, mas há outras ativações para introduzir não linearidade em seu modelo.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


########## nn.Sequential ##########

    # nn.Sequential é um contêiner ordenado de módulos. Os dados são passados por todos os módulos na mesma ordem definida. Você pode usar contêineres sequenciais para montar uma rede rápida como seq_modules. 

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)



########## nn.Softmax ##########


    # A última camada linear da rede neural retorna logits - valores brutos em [-infty, infty] - que são passados para o módulo nn.Softmax. Os logits são dimensionados para valores [0, 1] representando as probabilidades previstas do modelo para cada classe. O parâmetro dim indica a dimensão ao longo da qual os valores devem somar 1. 

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


########## Parâmetros do modelo ##########

    # Muitas camadas dentro de uma rede neural são parametrizadas, ou seja, possuem pesos e vieses associados que são otimizados durante o treinamento. A subclasse de nn.Module rastreia automaticamente todos os campos definidos dentro de seu objeto de modelo e torna todos os parâmetros acessíveis usando os métodos parameters() ou named_parameters() do seu modelo.

    # Neste exemplo, iteramos sobre cada parâmetro e imprimimos seu tamanho e uma visualização de seus valores. 





print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


########## Leitura adicional ##########

    # https://pytorch.org/docs/stable/nn.html