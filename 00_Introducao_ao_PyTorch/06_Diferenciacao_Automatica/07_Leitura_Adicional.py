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


########## Desativando o rastreamento de gradiente ##########


    # Por padrão, todos os tensores com require_grad=True estão rastreando seu histórico computacional e suportam computação de gradiente. No entanto, existem alguns casos em que não precisamos fazer isso, por exemplo, quando treinamos o modelo e queremos apenas aplicá-lo a alguns dados de entrada, ou seja, queremos apenas fazer cálculos diretos pela rede. Podemos parar de rastrear cálculos envolvendo nosso código de computação com o bloco torch.no_grad(): 



z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

    # Outra maneira de obter o mesmo resultado é usar o método detach() no tensor: 


z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)


    # Existem motivos para você querer desabilitar o rastreamento de gradiente:

        # Para marcar alguns parâmetros em sua rede neural como parâmetros congelados. Este é um cenário muito comum para ajustar uma rede pré-treinada

        # Para acelerar os cálculos quando você está apenas fazendo passagem direta, porque os cálculos em tensores que não rastreiam gradientes seriam mais eficientes. 


########## Mais sobre gráficos computacionais ##########


    # Conceitualmente, o autograd mantém um registro de dados (tensores) e todas as operações executadas (junto com os novos tensores resultantes) em um gráfico acíclico direcionado (DAG) consistindo em objetos Function. Neste DAG, as folhas são os tensores de entrada, as raízes são os tensores de saída. Ao traçar este gráfico das raízes às folhas, você pode calcular automaticamente os gradientes usando a regra da cadeia.

    # Em um passe para frente, o autograd faz duas coisas simultaneamente: 

        # execute a operação solicitada para calcular um tensor resultante

        # manter a função gradiente da operação no DAG 

    # A passagem para trás começa quando .backward() é chamado na raiz do DAG. autograd então: 


        # calcula os gradientes de cada .grad_fn,

        # os acumula no atributo .grad do respectivo tensor

        # usando a regra da cadeia, propaga todo o caminho para os tensores folha. 


    # NOTA

        # DAGs são dinâmicos no PyTorch Uma coisa importante a ser observada é que o gráfico é recriado do zero; após cada chamada .backward(), o autograd começa a preencher um novo gráfico. Isso é exatamente o que permite que você use instruções de fluxo de controle em seu modelo; você pode alterar a forma, o tamanho e as operações a cada iteração, se necessário. 



########## Leitura opcional: gradientes tensores e produtos jacobianos ##########

    # Em muitos casos, temos uma função de perda escalar e precisamos calcular o gradiente em relação a alguns parâmetros. No entanto, há casos em que a função de saída é um tensor arbitrário. Nesse caso, o PyTorch permite calcular o chamado produto Jacobiano, e não o gradiente real. 
    

    # Para uma função vetorial \vec{y}=f(\vec{x}) y​ = f(x ), onde \vec{x}=\langle x_1,\dots,x_n\rangle x =⟨x 1​ , …,xn​ ⟩ e \vec{y}=\langle _1,\dots,y_m\rangle y​ =⟨y 1 ,…,ym⟩, um gradiente de \vec{y} y em relação a \vec{x } x é dado pela matriz Jacobiana: 



    # Em vez de calcular a própria matriz Jacobiana, o PyTorch permite calcular o Produto Jacobiano v^T\cdot Jv T ⋅J para um determinado vetor de entrada v=(v_1 \dots v_m)v=(v 1 …v m ). Isso é conseguido chamando para trás com vv como argumento. O tamanho de vv deve ser igual ao tamanho do tensor original, em relação ao qual queremos calcular o produto: 

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

    # Observe que quando chamamos para trás pela segunda vez com o mesmo argumento, o valor do gradiente é diferente. Isso acontece porque ao fazer a propagação para trás, o PyTorch acumula os gradientes, ou seja, o valor dos gradientes computados é adicionado à propriedade grad de todos os nós folha do gráfico computacional. Se você quiser calcular os gradientes adequados, você precisa zerar a propriedade grad antes. No treinamento da vida real, um otimizador nos ajuda a fazer isso. 


    # NOTA

        # Anteriormente estávamos chamando a função backward() sem parâmetros. Isso é essencialmente equivalente a chamar backward(torch.tensor(1.0)), que é uma maneira útil de calcular os gradientes no caso de uma função de valor escalar, como perda durante o treinamento da rede neural. 


########## Leitura adicional  ##########

    # https://pytorch.org/docs/stable/notes/autograd.html