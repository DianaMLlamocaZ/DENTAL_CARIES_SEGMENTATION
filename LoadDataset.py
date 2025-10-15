import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from ReadImg import get_dataset


#Path main
path_main_carp="./dataset"


#Obtengo los dataset objects (train y test)
dataset_train,dataset_test=get_dataset(path_main=path_main_carp,train_size=80,augment_train=True,augment_test=False)


#Para crear los batches y entrenar la red creo un dataloader
dataloader_train=DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
dataloader_test=DataLoader(dataset=dataset_train,batch_size=32,shuffle=False)


#Entrenamiento
for index,data_batch in enumerate(dataloader_train):

    imgs_batch,masks_batch=data_batch #desempaqueto las imágenes y máscaras por bacth (#batch_samples,canal,alto,ancho) listas para entrenar

    print(index,imgs_batch.shape,masks_batch.shape)    
    