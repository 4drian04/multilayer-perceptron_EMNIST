# 🧠 Analizador Alfanumérico con Perceptrón Multicapa (EMNIST)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Made with](https://img.shields.io/badge/Made%20with-TensorFlow%20%26%20Keras-ff69b4)
![Dataset](https://img.shields.io/badge/Dataset-EMNIST-blueviolet)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen)


---

Proyecto de Inteligencia Artificial basado en redes neuronales profundas para el reconocimiento de caracteres alfanuméricos (dígitos y letras) utilizando el dataset EMNIST/bymerge.

---

## 📌 Descripción del proyecto

Este proyecto implementa un Perceptrón Multicapa (MLP) entrenado con TensorFlow/Keras capaz de clasificar imágenes de caracteres manuscritos.

El modelo ha sido diseñado para reconocer 47 clases (números + letras), alcanzando una precisión cercana al 88%, con buena capacidad de generalización.

---

## 🗂️ Dataset

Se utiliza el dataset EMNIST/bymerge, que:

- Contiene 814,255 imágenes
- 697,932 → entrenamiento  
- 116,323 → test  
- Imágenes de 28x28 píxeles en escala de grises
- 47 clases (letras y números)

Diferencia clave:
- bymerge agrupa letras mayúsculas y minúsculas similares (ej: C y c) en una misma clase.

---

## ⚙️ Tecnologías utilizadas

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Matplotlib / Seaborn  
- Scikit-learn  
- Plotly  

---

## 🧪 Metodología

### 🔄 Preprocesamiento de datos

- Transposición de imágenes (corrección de orientación)
- Normalización de píxeles → rango [0,1]
- Aplanado de imágenes → vector de 784 elementos
- Codificación One-Hot de etiquetas

---

### 🧠 Arquitectura del modelo

Mejor modelo obtenido:

Entrada: 784 neuronas  
↓  
Dense (256, ReLU)  
↓  
Dense (256, ReLU)  
↓  
Salida (47, Softmax)  

- Optimizador: Adam  
- Learning rate: 0.001  
- Función de pérdida: categorical_crossentropy  
- EarlyStopping para evitar overfitting  

---

### 🔍 Búsqueda de hiperparámetros

Se realizó un Grid Search manual probando:

- Capas: 1 y 2  
- Neuronas: 128, 256  
- Activaciones: ReLU, Tanh, Sigmoid  
- Optimizadores: Adam, SGD, RMSprop  
- Learning rates: 0.001, 0.01  

Conclusiones:

- Adam fue el mejor optimizador en la mayoría de los casos  
- ReLU + 0.001 obtuvo los mejores resultados  
- Dos capas mejoran ligeramente el rendimiento  

---

## 📈 Resultados

- Accuracy final: ~88%  
- Buen rendimiento general  
- Errores principalmente entre caracteres visualmente similares  

Ejemplos:
- 1 ↔ I  
- 0 ↔ O  
- 9 ↔ q  

---

## 📊 Evaluación

Se incluyen:

- Matriz de confusión (normalizada y numérica)  
- Visualización de errores  
- Métrica Accuracy  
- Gráfico 3D (Plotly) del impacto de neuronas en el rendimiento  

---

## 🖼️ Predicción de imágenes externas

El modelo permite predecir imágenes propias siguiendo estos pasos:

1. Convertir a escala de grises  
2. Redimensionar a 28x28  
3. Normalizar  
4. Aplanar (784)  
5. Añadir dimensión batch  
6. Predecir  

---

## 📁 Estructura del proyecto

```
├── main.py
├── utils.py
├── Analizador_Alfanumerico_AGG.h5
├── mejores_resultados.csv
├── imágenes de prueba (ej: q.png)
```

---

## 🚀 Cómo ejecutar

Instalar dependencias:
  ```bash
  pip install tensorflow tensorflow-datasets pillow matplotlib numpy keras scikit-learn seaborn plotly pandas scipy
  ```

Ejecutar el proyecto
  ```bash
  python main.py
  ```


## 📌 Conclusiones

- El modelo generaliza correctamente  
- Los errores son coherentes (caracteres similares)  
- Adam demuestra ser el optimizador más robusto  
- No hay gran diferencia entre 1 y 2 capas, pero el mejor modelo es:

👉 2 capas de 256 neuronas  

---

## ⏱️ Coste computacional

- Entrenamiento completo: ~9 horas (búsqueda de hiperparámetros)  
- Convergencia típica: 2–3 épocas  
- Batch size: 256  

---

## 👨‍💻 Autor

**Adrián García García** - [LinkedIn](https://www.linkedin.com/in/adri%C3%A1n-garc%C3%ADa-garc%C3%ADa-6ab399333/)
