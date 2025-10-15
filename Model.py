#Importo la libería necesaria para obtener el modelo preentrenado
import segmentation_models_pytorch as smp
from torch import nn
import torch


from torch.utils.data import DataLoader

#Instancio el modelo
modelo=smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",
                in_channels=1,classes=1,activation=None) #Se obtienen logits crudos


#Entrenamiento solo del decoder --> El encoder NO se entrenará (requires_grad=False) 
for params in modelo.encoder.parameters():
    params.requires_grad=False



#===

#DATOOOOS
from ReadImg import get_dataset


#Path main
path_main_carp="./dataset"


#Obtengo los dataset objects (train y test)
dataset_train,dataset_test=get_dataset(path_main=path_main_carp,train_size=80)


#Para crear los batches y entrenar la red creo un dataloader
dataloader_train=DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
dataloader_test=DataLoader(dataset=dataset_train,batch_size=32,shuffle=False)

#===


#Optimizer donde SOLO se entrenarán los pesos con requires_grad=True
optimizer=torch.optim.Adam(params=filter(lambda p:p.requires_grad,modelo.parameters()),lr=1e-3) #0.001 = 1x10''(-3)

#Hiperparámetros
epocas=100

#Losses
loss=torch.nn.BCEWithLogitsLoss()


for epoca in range(epocas):
    error_epoca=0

    for index,data_batch in enumerate(dataloader_train):
        imgs_batch,masks_batch=data_batch #Imágenes y máscaras por batch

        masks_pred=modelo(imgs_batch) #Aquí se tienen las máscaras predichas por el modelo para ese batch (matrices con logits crudos)

        error=loss(masks_pred,masks_batch) #Calculo el error


        optimizer.zero_grad() #Cero los gradientes
        error.backward() #Calculo los gradientes en base al error
        optimizer.step() #Actualizo los pesos

        
        error_epoca+=error #Error por época

    print(f"Época {index}. Loss: {error_epoca}")


 #¿Cómo se sobreentiende que tengo que usar BCEWithLoss en una matriz y no en un vector? :O Tema a ver
       

        