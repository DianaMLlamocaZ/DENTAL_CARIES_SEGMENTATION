# DENTAL_CARIES_SEGMENTATION
Este proyecto se basa en la creación/implementación de un modelo de segmentación de caries en radiografías dentales. 

# ARCHIVOS
- **Dataset.py**: Contiene el código para crear el custom dataset (recibe listas de imágenes y máscaras). Además, contiene el "data augmentation" dependiendo del parámetro "augment".
  
- **LoadDataset.py**: Contiene el código para cear el dataloader y verificación de datos por cada batch.
  
- **Model.py**: Código *referencial* del entrenamiento, usando modelos de arquitectura U-Net de la librería **segmentation_models_pytorch** con encoder de pesos preentrenados.

  **IMPORTANTE: Usé Kaggle para entrenar al modelo y utilizar la GPU. Por lo que el código del entrenamiento se encuentra en un notebook creado en Kaggle**
 
- **ReadImg.py**:
  -  Cargar cada imagen y máscara individualmente desde su path.
  -  Juntar las imágenes y máscaras en listas para realizar la división de datos en train/val sets mediante slicing.
  -  Crear el dataset object utilizando el custom dataset definido anteriormente.
  -  Uso de patches por imagen, ya que las imágenes son, en promedio, de 1000x1000.
    
- **losses_metrics**:
  - Este archivo contiene las losses que se usarán para entrenar el modelo. 
