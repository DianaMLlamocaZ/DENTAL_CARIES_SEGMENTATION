#Aquí se van a crear las losses y métricas para entrenar y evaluar el modelo :)
import torch

#LOSSES:
#BCELoss
bce_loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([8]).to(device)) #Coloco un WEIGHT más alto para la imbalanced class

#DiceLoss
def dice_loss(pred,target,smooth=1):
    #Se aplica 'sigmoid' para convertir a probabilidades los logits
    pred=torch.sigmoid(pred)

    #Calculo la intersección
    intersection=(pred*target).sum(dim=(2,3))

    #Calculo la 'unión' (pred + targets)
    union=pred.sum(dim=(2,3))+target.sum(dim=(2,3))

    #Calculo el Dice Coefficiente
    dice_coef=(2.*intersection+smooth)/(union+smooth)

    #Return el dice loss: 1-dice_coef
    return 1 - dice_coef.mean()

#Dice+BCE
def dice_bce(m_pred,m_target,smooth=1):
    #BCE ya lo definí anteriormente, así que solo lo uso para obtener el valor del loss
    loss_value_bce=bce_loss(m_pred,m_target)

    #DiceLoss, solamente uso la loss que definí
    loss_value_dice=dice_loss_smp(m_pred,m_target)

    #Loss FINAL: bce+dice
    loss_final=loss_value_bce+loss_value_dice

    return loss_final

#FocalTverskyLoss
def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=0.75, smooth=1):
    pred = torch.sigmoid(pred)

    TP = (pred * target).sum(dim=(2, 3))
    FP = (pred * (1 - target)).sum(dim=(2, 3))
    FN = ((1 - pred) * target).sum(dim=(2, 3))

    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    focal_tversky = torch.pow((1 - tversky_coef), gamma)

    return focal_tversky.mean()

    return loss_final


#MÉTRICAS:

#Precision:
def precision(mask_pred_log,mask_org,threshold):
  #Convierto la máscara predicha con logits a máscaras con píxeles de 0-1
  sm=torch.nn.Sigmoid()
  mask_pred_probs=sm(mask_pred_log)
  
  #Máscara predicha binarizada (0 y 1) de acuerdo al threshold
  mask_pred_th=(mask_pred_probs>threshold).float()

  #Calculo la cantidad de verdaderos positivos en la imagen: INTERSECCIÓN
  t_p=(mask_pred_th*mask_org).sum()
  
  #Calculo TP+FP
  tp_fp=(mask_pred_th).sum()
  
  #Precision
  precision=t_p/tp_fp

  return precision


#Recall:
def recall(mask_pred_log,mask_org,threshold):
  #Sigmoid
  sgm=torch.nn.Sigmoid()

  #Máscara: logs -> probs
  mask_pred_probs=sgm(mask_pred_log)

  #Máscara threshold
  mask_pred_th=(mask_pred_probs>threshold).float()

  #True positive
  tp=(mask_pred_th*mask_org).sum()

  #TP+FN
  tp_fn=(mask_org).sum()

  #Recall
  recall=tp/tp_fn

  return recall
