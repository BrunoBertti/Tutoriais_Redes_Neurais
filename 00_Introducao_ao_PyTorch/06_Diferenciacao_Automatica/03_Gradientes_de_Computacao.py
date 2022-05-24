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



########## Tensores, Funções e Gráfico Computacional ##########

    # Este código define o seguinte gráfico computacional: 

        # https://pytorch.org/tutorials/_images/comp-graph.png

    # Nesta rede, w e b são parâmetros, que precisamos otimizar. Assim, precisamos ser capazes de calcular os gradientes da função de perda em relação a essas variáveis. Para fazer isso, definimos a propriedade require_grad desses tensores. 

    # Nota: Você pode definir o valor de require_grad ao criar um tensor ou posteriormente usando o método x.requires_grad_(True). 


    # Uma função que aplicamos a tensores para construir grafos computacionais é de fato um objeto da classe Function. Este objeto sabe como calcular a função na direção para frente e também como calcular sua derivada durante a etapa de propagação para trás. Uma referência à função de propagação para trás é armazenada na propriedade grad_fn de um tensor. Você pode encontrar mais informações sobre Função na documentação. 

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


########## Gradientes de computaçãol ##########

    # Para otimizar os pesos dos parâmetros na rede neural, precisamos calcular as derivadas de nossa função de perda em relação aos parâmetros, ou seja, precisamos de ∂loss/∂w e ∂loss/∂b sob alguns valores fixos de x e y. Para calcular essas derivadas, chamamos loss.backward() e, em seguida, recuperamos os valores de w.grad e b.grad: 


loss.backward()
print(w.grad)
print(b.grad)


    # NOTA:

        # Só podemos obter as propriedades grad para os nós da folha do gráfico computacional, que possuem a propriedade require_grad configurada para True. Para todos os outros nós em nosso gráfico, os gradientes não estarão disponíveis.

        # Só podemos realizar cálculos de gradiente usando para trás uma vez em um determinado gráfico, por motivos de desempenho. Se precisarmos fazer várias chamadas inversas no mesmo gráfico, precisamos passar keep_graph=True para a chamada inversa. 