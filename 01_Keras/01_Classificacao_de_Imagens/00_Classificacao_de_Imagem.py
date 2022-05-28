###### Esse código foi retirado de https://www.tensorflow.org/tutorials/images/classification#import_tensorflow_and_other_libraries




########## Classificação de Imagens ##########  


    # Este tutorial mostra como classificar imagens de flores. Ele cria um classificador de imagem usando um modelo tf.keras.Sequential e carrega dados usando tf.keras.utils.image_dataset_from_directory . Você ganhará experiência prática com os seguintes conceitos:

        # Carregar com eficiência um conjunto de dados fora do disco.

        # Identificar overfitting e aplicar técnicas para mitigá-lo, incluindo aumento e abandono de dados.

    # Este tutorial segue um fluxo de trabalho básico de aprendizado de máquina:


        
        # 01 - Examinar e entender os dados
        
        # 02 - Construir um pipeline de entrada
        
        # 03 - Construir o modelo
        
        # 04 - Treine o modelo
        
        # 05 - Teste o modelo
        
        # 06 - Melhore o modelo e repita o processo



