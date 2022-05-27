########## SALVAR E CARREGAR O MODELO ##########

    # Nesta seção, veremos como persistir o estado do modelo salvando, carregando e executando as previsões do modelo.


import torch
import torchvision.models as models




########## Salvando e carregando pesos de modelo ##########


    # Os modelos PyTorch armazenam os parâmetros aprendidos em um dicionário de estado interno, chamado state_dict. Eles podem ser persistidos através do método torch.save: 

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')


    # Para carregar pesos de modelo, você precisa primeiro criar uma instância do mesmo modelo e, em seguida, carregar os parâmetros usando o método load_state_dict(). 

model = models.vgg16() # não especificamos pretrained=True, ou seja, não carregamos pesos padrão 
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()



    # NOTA:

        # certifique-se de chamar o método model.eval() antes de inferir para definir as camadas de eliminação e normalização de lote para o modo de avaliação. Deixar de fazer isso produzirá resultados de inferência inconsistentes. 