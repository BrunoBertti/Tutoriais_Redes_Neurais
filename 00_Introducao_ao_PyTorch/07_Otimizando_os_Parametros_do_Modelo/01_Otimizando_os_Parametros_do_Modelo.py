
########## OTIMIZAÇÃO DOS PARÂMETROS DO MODELO ##########



    # Agora que temos um modelo e dados, é hora de treinar, validar e testar nosso modelo otimizando seus parâmetros em nossos dados. O treinamento de um modelo é um processo iterativo; em cada iteração (chamada de época) o modelo faz uma estimativa sobre a saída, calcula o erro em sua estimativa (perda), coleta as derivadas do erro em relação aos seus parâmetros (como vimos na seção anterior) e otimiza esses parâmetros usando gradiente descendente. Para um passo a passo mais detalhado desse processo, confira este vídeo sobre retropropagação de 3Blue1Brown. 