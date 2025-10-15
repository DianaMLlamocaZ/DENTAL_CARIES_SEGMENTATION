import os
import cv2
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import random

from Dataset import DentalSegmentationDataset
import torch

#Función para cargar cada imagen y mask en array
def load_img(path_main,name_img):
   path_img=path_main+f"/{name_img}.nrrd"
   path_mask=path_main+f"/{name_img}_mask.nrrd"

   img_stik=sitk.ReadImage(path_img)
   mask_stik=sitk.ReadImage(path_mask)

   tensor_img=sitk.GetArrayFromImage(img_stik)
   tensor_mask=sitk.GetArrayFromImage(mask_stik)

   #Si hay 4 canales (o más por la conversión) --> Obtengo 1 solo canal, ya que los 4 canales tienen los mismos valores de píxeles
   if tensor_img.ndim>=4:
      tensor_img_f=tensor_img[:,:,:,0]
      return tensor_img_f[0],tensor_mask[0]
   
   else:   
    return tensor_img[0],tensor_mask[0]



#Función para obtener la lista de imágenes y masks completas
def read_img(path_carp_img):

   imgs_stik=[]
   masks_stik=[]

   #cont=0 #verificar cuántos son iguales PRUEBA


   for name_carp in os.listdir(path_carp_img):
      path_complete_pac=path_carp_img+f"/{name_carp}" #path de la carpeta por paciente

      img,mask=load_img(path_complete_pac,name_carp)


      #Comprobar si son iguales los shapes PRUEBA
      #if img.shape==mask.shape: #PRUEBA
      #   cont+=1   #PRUEBA
      #else: #PRUEBA
      #   print(f"{name_carp} diferentes")  #PRUEBA

      imgs_stik.append(img),masks_stik.append(mask)
   
   #print("cantidades iguales:",cont) #PRUEBA

   return imgs_stik,masks_stik



#Train_test split función que recibe 2 parámetros: file path y el size de la data de train (% del total)
def train_test_split(path_main,train_size):
   imgs,masks=read_img(path_main)
   
   num_samples_train=int(len(imgs)*(train_size/100)) #número de samples en test

   train_imgs,train_masks=imgs[0:num_samples_train],masks[0:num_samples_train] #separación de datos de train y test
   test_imgs,test_masks=imgs[num_samples_train:len(imgs)],masks[num_samples_train:len(imgs)]

   
   return train_imgs,train_masks,test_imgs,test_masks #Son listas de arrays lo que se devuelve
   
#=======

#Diviendo las imágenes en patches: Realizar un down scale directamente de las imágenes a 256x256 puede afectar a la calidad de las imágenes
def extract_balanced_patches(imgs, masks, patch_size=256, stride=128, min_mask_area=100, neg_ratio=0.5, seed=42):
    random.seed(seed)
    pos_patches = []
    neg_patches = []

   #Creando los patches por cada imagen
    for img, mask in zip(imgs, masks):
        H, W = img.shape
        for top in range(0, H - patch_size + 1, stride):
            for left in range(0, W - patch_size + 1, stride):
                img_patch = img[top:top + patch_size, left:left + patch_size]
                mask_patch = mask[top:top + patch_size, left:left + patch_size]

                if mask_patch.sum() >= min_mask_area:
                    pos_patches.append((img_patch, mask_patch))
                else:
                    neg_patches.append((img_patch, mask_patch))

    #Limitar los píxeles 'negativos' para balancear
    num_pos = len(pos_patches)
    num_neg = int(num_pos * neg_ratio)
    neg_patches = random.sample(neg_patches, min(len(neg_patches), num_neg))

    all_patches = pos_patches + neg_patches
    random.shuffle(all_patches)

    patch_imgs, patch_masks = zip(*all_patches)
    return list(patch_imgs), list(patch_masks)

#==========
   

#Función para obtener el dataset object
def get_dataset(path_main,train_size,augment_train,augment_test):
   
   imgs_train,masks_train,imgs_test,masks_test=train_test_split(path_main,train_size) #listas y máscaras de train y test

   #==
   #Llamo a la función para crear patches: obtengo listas de patches
   patch_imgs_train,patch_masks_train=extract_balanced_patches(imgs_train,masks_train,stride=128,min_mask_area=10,neg_ratio=0.5)
   patch_imgs_test,patch_masks_test=extract_balanced_patches(imgs_test,masks_test,stride=128,min_mask_area=10,neg_ratio=0.5)
   #==

   dataset_train=DentalSegmentationDataset(list_imgs=patch_imgs_train,list_masks=patch_masks_train,augment=augment_train) #list_imgs=imgs_train
   dataset_test=DentalSegmentationDataset(list_imgs=patch_imgs_test,list_masks=patch_masks_test,augment=augment_test) #list_imgs=imgs_test
   

   return dataset_train,dataset_test #Retorno los Dataset para ambos conjuntos de estos


#train_test_split("./dataset",train_size=80)
#ds_tr,ds_te=get_dataset("./dataset",train_size=80,augment_train=True,augment_test=False)
#img1=ds_tr[1] #img index 1
#Cargo la data
#read_img(path_main_carp)

"""
cont=0
for i in range(len(os.listdir(path_main_carp))):
   cont+=1
   
   print(f"Imagen {cont}: Shape img: {imgs_stik[i].shape}, Shape mask: {masks_stik[i].shape}")



from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

ds=DentalSegmentationDataset(imgs_stik,masks_stik,augment=True)
dataset=next(iter(ds))
img,mask=dataset[0],dataset[1]
plt.imshow(img.reshape(img.shape[1],img.shape[2]),cmap="gray")
plt.show()
plt.imshow(mask.reshape(img.shape[1],img.shape[2]),cmap="gray")
plt.show()

dataloader=DataLoader(ds,batch_size=2) #3 imágenes por batch
print(len(dataloader))

cont=0
for img,mask in dataloader: #Itero por cada batch
   print(img.shape) #tamaño del batch == cantidad de imágenes en el batch --> representado por img.shape[0] = dim en el eje inicial
   #print(mask.shape)
   for i in range(img.shape[0]): #Aquí itero sobre cada una de las imágenes contenidas en el batch
      print(img[i].shape)
      plt.imshow(img[i].reshape(img[i].shape[1],img[i].shape[2]),cmap="gray")
      plt.show()
      plt.imshow(mask[i].reshape(img[i].shape[1],img[i].shape[2]),cmap="gray")
      plt.show()

"""
