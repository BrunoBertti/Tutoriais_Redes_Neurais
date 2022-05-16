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