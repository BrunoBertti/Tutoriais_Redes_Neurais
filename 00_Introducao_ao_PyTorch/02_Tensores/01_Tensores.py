########## TENSORES ##########

    # Tensores são uma estrutura de dados especializada que são muito semelhantes a arrays e matrizes. No PyTorch, usamos tensores para codificar as entradas e saídas de um modelo, bem como os parâmetros do modelo.

    # Os tensores são semelhantes aos ndarrays do NumPy, exceto que os tensores podem ser executados em GPUs ou outros aceleradores de hardware. Na verdade, tensores e matrizes NumPy podem frequentemente compartilhar a mesma memória subjacente, eliminando a necessidade de copiar dados (consulte Ponte com NumPy). Os tensores também são otimizados para diferenciação automática (veremos mais sobre isso posteriormente na seção Autograd). Se você estiver familiarizado com ndarrays, estará em casa com a API do Tensor. Se não, acompanhe! 


import torch
import numpy as np


