
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



########## Hiperparâmetros ##########

    # Os hiperparâmetros são parâmetros ajustáveis que permitem controlar o processo de otimização do modelo. Valores de hiperparâmetros diferentes podem afetar o treinamento do modelo e as taxas de convergência (leia mais sobre o ajuste de hiperparâmetros)

    # Definimos os seguintes hiperparâmetros para treinamento:


        # Número de épocas - o número de vezes para iterar no conjunto de dados

        # Tamanho do lote - o número de amostras de dados propagadas pela rede antes que os parâmetros sejam atualizados

        # Taxa de aprendizado - quanto atualizar os parâmetros dos modelos em cada lote/época. Valores menores geram uma velocidade de aprendizado lenta, enquanto valores grandes podem resultar em comportamento imprevisível durante o treinamento. 



learning_rate = 1e-3
batch_size = 64
epochs = 5



########## Ciclo de otimização ##########


    # Depois de definir nossos hiperparâmetros, podemos treinar e otimizar nosso modelo com um loop de otimização. Cada iteração do loop de otimização é chamada de época.

    # Cada época consiste em duas partes principais:


        # O Train Loop - itere sobre o conjunto de dados de treinamento e tente convergir para os parâmetros ideais.

        # O loop de validação/teste - itere sobre o conjunto de dados de teste para verificar se o desempenho do modelo está melhorando.

    # Vamos nos familiarizar brevemente com alguns dos conceitos usados no loop de treinamento. Avance para ver a implementação completa do loop de otimização. 


########## Função de perda ##########

    # Quando apresentados a alguns dados de treinamento, nossa rede não treinada provavelmente não fornecerá a resposta correta. A função de perda mede o grau de dissimilaridade do resultado obtido em relação ao valor alvo, e é a função de perda que queremos minimizar durante o treinamento. Para calcular a perda, fazemos uma previsão usando as entradas de nossa amostra de dados fornecida e a comparamos com o valor real do rótulo de dados.

    # As funções de perda comuns incluem nn.MSELoss (Mean Square Error) para tarefas de regressão e nn.NLLLoss (Negative Log Likelihood) para classificação. nn.CrossEntropyLoss combina nn.LogSoftmax e nn.NLLLoss.

    # Passamos os logits de saída do nosso modelo para nn.CrossEntropyLoss, que normalizará os logits e calculará o erro de previsão. 


# Inicialize a função de perda 
loss_fn = nn.CrossEntropyLoss()



########## Otimizador ##########


    # A otimização é o processo de ajustar os parâmetros do modelo para reduzir o erro do modelo em cada etapa de treinamento. Algoritmos de otimização definem como esse processo é realizado (neste exemplo, usamos Stochastic Gradient Descent). Toda a lógica de otimização é encapsulada no objeto otimizador. Aqui, usamos o otimizador SGD; além disso, existem muitos otimizadores diferentes disponíveis no PyTorch, como ADAM e RMSProp, que funcionam melhor para diferentes tipos de modelos e dados.

    # Inicializamos o otimizador registrando os parâmetros do modelo que precisam ser treinados e passando o hiperparâmetro de taxa de aprendizado. 


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # Dentro do loop de treinamento, a otimização acontece em três etapas:

        # Chame optimizer.zero_grad() para redefinir os gradientes dos parâmetros do modelo. Gradientes por padrão se somam; para evitar a contagem dupla, zeramos explicitamente a cada iteração.

        # Retropropague a perda de previsão com uma chamada para loss.backward(). PyTorch deposita os gradientes da perda w.r.t. cada parâmetro.

        # Uma vez que temos nossos gradientes, chamamos optimizer.step() para ajustar os parâmetros pelos gradientes coletados na passagem para trás. 


########## Implementação completa ##########


    # Definimos train_loop que faz um loop sobre nosso código de otimização e test_loop que avalia o desempenho do modelo em relação aos nossos dados de teste. 


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Calcular previsão e perda 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



    # Inicializamos a função de perda e o otimizador e passamos para train_loop e test_loop. Sinta-se à vontade para aumentar o número de épocas para acompanhar a melhoria do desempenho do modelo. 


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")