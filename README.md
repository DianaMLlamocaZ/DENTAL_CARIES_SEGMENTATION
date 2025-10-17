# DENTAL_CARIES_SEGMENTATION
Este proyecto se basa en la creación/implementación de un modelo de segmentación de caries en radiografías dentales. 

# ARCHIVOS
- [**Dataset.py**](./Dataset.py): Contiene el código para crear el custom dataset (recibe listas de imágenes y máscaras). Además, contiene el "data augmentation" dependiendo del parámetro "augment".
  
- [**LoadDataset.py**](./LoadDataset.py): Contiene el código para crear el dataloader y verificar los datos por cada batch.
  
- [**Model.py**](./Model.py): Código que contiene un modelo preentrenado de Segmentation Models Ptorch y un modelo U-Net small creado desde cero para realizar pruebas debido a la cantidad de datos. Además, contiene un pequeño código *referencial* del entrenamiento, usando modelos de arquitectura U-Net de la librería **segmentation_models_pytorch** con encoder de pesos preentrenados.

  **IMPORTANTE: Usé Kaggle para entrenar al modelo y utilizar la GPU. Por lo que el código del entrenamiento se encuentra en un notebook creado en Kaggle**
 
- [**ReadImg.py**](./ReadImg.py):
  -  Cargar cada imagen y máscara individualmente desde su path.
  -  Juntar las imágenes y máscaras en listas para realizar la división de datos en train/val sets mediante slicing.
  -  Crear el dataset object utilizando el custom dataset definido anteriormente.
  -  Uso de patches por imagen, ya que las imágenes son, en promedio, de 1000x1000.
    
- [**losses_metrics**](./losses_metrics.py):
  - Este archivo contiene las losses y métricas que se usarán para entrenar y evaluar el performance del modelo.
 
# IMPORTANTE
- El entrenamiento del modelo se está realizando en Kaggle por el uso de la GPU. Estos archivos fueron creados para utilizar las funciones definidas y entrenar el modelo directamente en el notebook de Kaggle, facilitando la carga de datos, del modelo, e implementación de las loss functions.
