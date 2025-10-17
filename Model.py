#Importo la libería necesaria para obtener el modelo preentrenado
import segmentation_models_pytorch as smp
from torch import nn
import torch


from torch.utils.data import DataLoader

#====

# 1) MODELOS
#A) SEGMENTATION MODELS PYTORCH
#Instancio el modelo
modelo=smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",
                in_channels=1,classes=1,activation=None) #Se obtienen logits crudos

#Entrenamiento solo del decoder --> El encoder NO se entrenará (requires_grad=False) 
for params in modelo.encoder.parameters():
    params.requires_grad=False

#=====

#B) CREACIÓN DE UN MODELO U-NET PEQUEÑO (4 LAYERS: ENC Y DECODER) DESDE CERO PARA PRUEBAS DE RENDIMIENTO Y POR LA CANTIDAD DE DATOS DISPONIBLES
class SkipC_CNN(torch.nn.Module):
  def __init__(self,in_channels):
    super().__init__()

    #ENCODER
    self.cnn1=torch.nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,padding=1)
    self.r1=torch.nn.ReLU()
    self.mp1=torch.nn.MaxPool2d(kernel_size=2) #Reduce a la mitad

    self.cnn2=torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
    self.r2=torch.nn.ReLU()
    self.mp2=torch.nn.MaxPool2d(kernel_size=2) #Reduce a la mitad

    self.cnn3=torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
    self.r3=torch.nn.ReLU()
    self.mp3=torch.nn.MaxPool2d(kernel_size=2) #Reduce a la mitad

    self.cnn4=torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
    self.r4=torch.nn.ReLU()
    self.mp4=torch.nn.MaxPool2d(kernel_size=2) #Reduce a la mitad


    #DECODER
    self.tc1=torch.nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=2,stride=2)
    self.r5=torch.nn.ReLU()
    self.sc1=torch.nn.Conv2d(in_channels=128+128,out_channels=128,kernel_size=3,padding=1) #El output size es igual [32,128,]

    self.tc2=torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=2,stride=2)
    self.r6=torch.nn.ReLU()
    self.sc2=torch.nn.Conv2d(in_channels=64+64,out_channels=64,kernel_size=3,padding=1) #El output size es igual [32,64,]

    self.tc3=torch.nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2)
    self.r7=torch.nn.ReLU()
    self.sc3=torch.nn.Conv2d(in_channels=32+32,out_channels=32,kernel_size=3,padding=1) #Output size [32,32,]

    self.tc4=torch.nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=2,stride=2)
    self.sgm=torch.nn.Sigmoid()


  def forward(self,data):
    #Encoder
    enc1=self.mp1(self.r1(self.cnn1(data))) #capas 1 --> [32,32,256,256]  INPUT 512X512
    print(f"res shape capa 1 enc: {enc1.shape}")

    enc2=self.mp2(self.r2(self.cnn2(enc1))) #capas 2 --> [32,64,128,128]
    print(f"res shape capa 2 enc: {enc2.shape}")

    enc3=self.mp3(self.r3(self.cnn3(enc2))) #capas 3 --> [32,128,64,64]
    print(f"res shape capa 3 enc: {enc3.shape}")

    enc4=self.mp4(self.r4(self.cnn4(enc3))) #capas 4 --> [32,256,32,32]
    print(f"res shape capa 4 enc: {enc4.shape}")


    #Decoder
    dec1=self.r5(self.tc1(enc4)) #capas 1 [32,128,64,64]
    #Skip connections
    sc1_vector=torch.cat([enc3,dec1],dim=1)
    sc1_result=self.sc1(sc1_vector) #Acá se pasan los 2 vectores a la conv. layer connection [32,128,64,64]
    print(f"res shape capa 1 dec: {sc1_result.shape}")


    dec2=self.r6(self.tc2(sc1_result)) #capas 2 [32,64,64,64]
    #Skip connections
    sc2_vector=torch.cat([enc2,dec2],dim=1)
    sc2_result=self.sc2(sc2_vector) #Skip connection
    print(f"res shape capa 2 dec: {sc2_result.shape}")


    dec3=self.r7(self.tc3(sc2_result)) #capas 3 [32,32,256,256]
    #Skip connections
    sc3_vector=torch.cat([enc1,dec3],dim=1)
    sc3_result=self.sc3(sc3_vector) #Skip connection
    print(f"res shape capa 3 dec: {sc3_result.shape}")


    #dec4=self.sgm(self.tc4(sc3_result)) #capas 4 [32,1,256,256]
    dec4=self.tc4(sc3_result) #Logits [32,1,256,256]
    print(f"res shape capa 4 dec: {dec4.shape}")

    return dec4

#====


#2) PRUEBA DEL MODELO SMP EN EL DATALOADER

#A) OBTENGO LOS DATOS Y CREACIÓN DEL DATALOADER
from ReadImg import get_dataset

#Path main
path_main_carp="./dataset"


#Obtengo los dataset objects (train y test)
dataset_train,dataset_test=get_dataset(path_main=path_main_carp,train_size=80)


#Para crear los batches y entrenar la red creo un dataloader
dataloader_train=DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
dataloader_test=DataLoader(dataset=dataset_train,batch_size=32,shuffle=False)



#B) HIPERPARÁMETROS 

#Optimizer donde SOLO se entrenarán los pesos con requires_grad=True
optimizer=torch.optim.Adam(params=filter(lambda p:p.requires_grad,modelo.parameters()),lr=1e-3) #0.001 = 1x10''(-3)

#Hiperparámetros
epocas=100

#Losses
loss=torch.nn.BCEWithLogitsLoss()



#C) TRAINING

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



        

