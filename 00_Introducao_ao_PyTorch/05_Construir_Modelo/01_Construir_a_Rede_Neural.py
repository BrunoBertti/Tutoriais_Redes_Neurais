########## CONSTRUA A REDE NEURAL ##########


    # As redes neurais são compostas por camadas/módulos que realizam operações nos dados. O namespace torch.nn fornece todos os blocos de construção necessários para construir sua própria rede neural. Cada módulo no PyTorch subclasse o nn.Module. Uma rede neural é um módulo em si que consiste em outros módulos (camadas). Essa estrutura aninhada permite construir e gerenciar arquiteturas complexas facilmente.

    # Nas seções a seguir, construiremos uma rede neural para classificar imagens no conjunto de dados FashionMNIST. 

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms