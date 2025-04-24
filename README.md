Este proyecto implementa un modelo de clasificación de colores utilizando una red neuronal convolucional (CNN) desarrollada con TensorFlow y Keras. Permite entrenar el modelo con imágenes de colores específicos y luego clasificar colores en tiempo real utilizando la cámara web.



🧠 Modelo
El modelo es una CNN secuencial con la siguiente arquitectura:

    Conv2D: 32 filtros, tamaño de kernel 3x3, activación ReLU
    
    MaxPooling2D: Tamaño de pool 2x2
    
    Conv2D: 64 filtros, tamaño de kernel 3x3, activación ReLU
    
    MaxPooling2D: Tamaño de pool 2x2
    
    Flatten: Aplanamiento de las características
    
    Dense: 64 unidades, activación ReLU
    
    Dense: 2 unidades (para dos clases), activación Softmax

🧪 Entrenamiento
Épocas: 25

    Función de pérdida: categorical_crossentropy
    
    Optimizador: Adam
    
    Métricas: accuracy
    
    Aumento de datos: Rotación, desplazamiento, zoom, volteo horizontal y variación de brillo
    
    División de datos: 80% entrenamiento, 20% validación

Las imágenes deben estar organizadas en carpetas por clase dentro de un directorio llamado dataset/.

📷 Clasificación en Tiempo Real
El script classify_image.py utiliza la cámara web para capturar imágenes en tiempo real, las procesa y clasifica en una de las dos clases entrenadas, mostrando la predicción y la confianza en pantalla.

🚀 Requisitos
    Python 3.x
    TensorFlow
    Keras
    OpenCV
    NumPy
    Matplotlib

Instala las dependencias con:

pip install -r requirements.txt

📝 Uso
Entrenar el modelo:

    bash
    python src/train_model.py
Clasificar en tiempo real:

    bash
    python src/classify_image.py
📄 Notas
Asegúrate de que las imágenes estén correctamente organizadas en carpetas por clase dentro del directorio dataset/.

El modelo entrenado se guarda en la carpeta Trained_model/ como model.h5.

Presiona la tecla q para salir de la clasificación en tiempo real.
