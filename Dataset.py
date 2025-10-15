import torch
from torchvision import transforms
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

#Dataset object para instanciar cada DS (para train y test)
class DentalSegmentationDataset(Dataset):
    def __init__(self,list_imgs,list_masks,size=(256,256),augment=False):
        self.imgs=list_imgs
        self.masks=list_masks
        self.size=size
        self.augment=augment

        #Si se realiza data augmentation
        if self.augment:
            print("Dataset con augmentation")
            self.transform=A.Compose([
                #Resize
                #A.Resize(height=256,width=256), #512,768 --> Al realizar PATCHING, no es necesario resize

               
                #Transformaciones
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                
                
                #Transformación a tensor
                ToTensorV2()
            ])

        #De lo contrario, solo resize
        else:
            print("Dataset sin augmentation")
            self.transform=A.Compose([
                #A.Resize(height=256,width=256), #512,768 --> Al realizar PATCHING, no es necesario resize
                ToTensorV2()
            ])
    

    def __len__(self):
        return len(self.imgs)
    
    
    def __getitem__(self,idx):
        img=self.imgs[idx] 
        mask_=self.masks[idx]

        #print("img shape:",img.shape) #Me aseguro que albumentation recibe (H,W)
        #print("mask shape:",mask_.shape) #Me aseguro que albumentation reciba (H,W)

        #Escalamiento a rango de [0,1] (aún numpy array)
        image=img.astype("float32")/255  #Acá debe tener el mismo nombre del argumento (así pide albumentations)
        mask=mask_.astype("float32")  #Igual acá


        #Aplico las transformaciones (si se define así) y obtengo las imágenes y máscaras
        transformaciones=self.transform(image=image,mask=mask) 
        img_,mask_=transformaciones["image"],transformaciones["mask"]
        
        #print("img_ shape:",img_.shape)
        #print("mask_ shape:",mask_.shape)
        
        return img_,mask_.unsqueeze(0) #A la máscara, le añado una dimensión haciendo referencia al canal