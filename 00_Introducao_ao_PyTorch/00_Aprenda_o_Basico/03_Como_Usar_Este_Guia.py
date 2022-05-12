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