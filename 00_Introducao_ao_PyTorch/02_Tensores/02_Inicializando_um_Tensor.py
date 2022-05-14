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