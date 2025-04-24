import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def predict_image(image_path, model_path="Trained_model/model.h5", img_size=(100, 100)):
    # Cargar el modelo entrenado
    model = load_model(model_path)
    print(f"Modelo cargado desde {model_path}")

    # Cargar y preprocesar la imagen
    img = load_img(image_path, target_size=img_size)  # Carga la imagen y la redimensiona
    img_array = img_to_array(img)  # Convierte la imagen a array numpy
    img_array = img_array / 255.0  # Normaliza la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añade dimensión batch

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Obtener el mapeo de clases (asumiendo que se cargó el generador de entrenamiento)
    # Aquí debes definir las clases manualmente o cargar desde el generador si tienes acceso
    class_indices = {0: 'Clase_0', 1: 'Clase_1'}  # Cambia estos nombres según tus clases reales

    predicted_class = class_indices.get(predicted_class_index, "Clase desconocida")
    confidence = predictions[0][predicted_class_index]

    print(f"Predicción: {predicted_class} con confianza {confidence:.2f}")

    return predicted_class, confidence

if __name__ == "__main__":
    # Ruta de la imagen que quieres probar
    image_path = "ruta/a/tu/imagen.jpg"

    # Ejecutar la predicción
    predict_image(image_path)
