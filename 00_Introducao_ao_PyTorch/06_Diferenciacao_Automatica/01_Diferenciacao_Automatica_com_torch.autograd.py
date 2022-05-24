########## DIFERENCIAÇÃO AUTOMÁTICA COM TOCHA.AUTOGRAD ##########


    # Ao treinar redes neurais, o algoritmo mais usado é o backpropagation. Neste algoritmo, os parâmetros (pesos do modelo) são ajustados de acordo com o gradiente da função de perda em relação ao parâmetro dado.

    # Para calcular esses gradientes, o PyTorch possui um mecanismo de diferenciação embutido chamado torch.autograd. Ele suporta computação automática de gradiente para qualquer gráfico computacional.

    # Considere a rede neural de uma camada mais simples, com entrada x, parâmetros w e b, e alguma função de perda. Ele pode ser definido no PyTorch da seguinte maneira: 


import torch

x = torch.ones(5)  # insira tensor
y = torch.zeros(3)  # saida esperada
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)