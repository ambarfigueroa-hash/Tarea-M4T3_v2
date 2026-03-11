# Detección de Equipos de Construcción con YOLOv8
### MAIC – Tarea de Visión Computacional (M4T3) AMBAR MARIA FIGUEROA FIGARI

---

# 1. Descripción del Proyecto

Este proyecto implementa un modelo de **detección de objetos (Equipos pesados de construccion en obra civil) utilizando YOLOv8** para identificar maquinaria y vehículos utilizados en entornos de construcción.

El objetivo es demostrar el flujo completo de desarrollo de un sistema de visión computacional, incluyendo:

- Preparación y etiquetado de datos
- Entrenamiento del modelo
- Evaluación del desempeño
- Análisis de errores
- Generación de inferencias con el modelo entrenado

El modelo fue entrenado utilizando el framework **Ultralytics YOLOv8** en el entorno de **Google Colab**, empleando un dataset anotado y gestionado mediante **Roboflow**.

---

# 2. Problema a Resolver

En obras de construcción existen múltiples tipos de maquinaria pesada que deben ser monitoreadas para:

- control operativo
- seguridad en obra
- análisis de productividad
- gestión de equipos

La visión computacional permite automatizar la detección de estos equipos en imágenes o videos, facilitando la supervisión de operaciones en entornos de construcción.

Este proyecto explora el uso de **detección automática de maquinaria** mediante modelos de deep learning.

---

# 3. Dataset

El dataset fue preparado utilizando la plataforma **Roboflow**, donde se realizaron las anotaciones de los objetos presentes en las imágenes.

**Plataforma de dataset:** Roboflow  
**Proyecto:** `Tarea-M4T3_v2`

### Clases detectadas

El modelo fue entrenado para detectar las siguientes clases:

- Excavator
- Dozer
- Grader
- Loader
- Agricultural Tractor
- Compactor 10 ton
- Articulated Truck
- Articulated Tank Truck
- Conventional Truck
- Van
- Bus
- Person

Las anotaciones se realizaron utilizando **bounding boxes** ajustadas al objeto visible en cada imagen.

### División del dataset

El dataset fue dividido en:

- **Training set**
- **Validation set**
- **Test set**

El dataset fue exportado en formato **YOLOv8**.

---

# 4. Modelo Utilizado

El modelo fue entrenado utilizando la arquitectura **YOLOv8s** del framework Ultralytics.

### Configuración del entrenamiento

| Parámetro | Valor |
|----------|------|
Modelo | YOLOv8s |
Épocas | 30 |
Tamaño de imagen | 640 |
Framework | Ultralytics YOLOv8 |
Entorno de entrenamiento | Google Colab (GPU Tesla T4) |

---

# 5. Métricas del Modelo

Las métricas fueron calculadas utilizando el conjunto de validación.

| Métrica | Resultado |
|-------|------|
Precision | 0.53 |
Recall | 0.278 |
mAP50 | 0.406 |
mAP50-95 | 0.305 |

Las curvas de entrenamiento y evaluación pueden encontrarse en la carpeta:
results/curves/

---

# 6. Resultados

Este repositorio incluye los siguientes resultados del entrenamiento:

- Curvas de entrenamiento
- Matriz de confusión
- Ejemplos de predicciones del modelo

Ubicación de los archivos:
results/curves/results.png
results/curves/confusion_matrix.png

Ejemplos de detección:
results/evidence/

Estas imágenes muestran las predicciones generadas por el modelo entrenado.

## Curvas de entrenamiento

![Curvas de entrenamiento](results/curves/results.png)

Estas curvas muestran la evolución del entrenamiento del modelo, incluyendo precisión, recall y métricas de evaluación durante las épocas de entrenamiento.

---

## Matriz de confusión

![Matriz de confusión](results/curves/confusion_matrix.png)

La matriz de confusión permite visualizar cómo el modelo clasifica cada tipo de maquinaria y qué clases pueden confundirse entre sí.

---

## Ejemplo de detección del modelo

![Ejemplo de detección](results/evidence/val_batch0_pred.jpg)

La imagen anterior muestra un ejemplo de predicción realizada por el modelo YOLOv8 entrenado.
---

# 7. Análisis de Errores

Análisis de errores del modelo
Interpretación general del desempeño

El modelo de detección basado en YOLOv8s fue evaluado utilizando el conjunto de validación, obteniendo los siguientes resultados globales:

Precision: 0.53

Recall: 0.278

mAP50: 0.406

mAP50-95: 0.305

Estos resultados indican que el modelo tiene un desempeño moderado, con una capacidad razonable para identificar correctamente objetos detectados, pero con limitaciones para detectar todos los objetos presentes en las imágenes.

El valor de precision (0.53) sugiere que aproximadamente el 53% de las detecciones realizadas por el modelo son correctas.
Sin embargo, el recall (0.278) indica que el modelo solo detecta alrededor del 27.8% de los objetos reales presentes en las imágenes, lo que sugiere una cantidad considerable de falsos negativos.

Durante la evaluación del modelo se observaron algunas limitaciones.

### Falsos positivos

Algunos vehículos pueden ser detectados incorrectamente cuando existen oclusiones parciales o similitud visual entre tipos de maquinaria.

### Falsos negativos

Objetos pequeños o maquinaria ubicada a gran distancia pueden no ser detectados correctamente.

### Análisis por clases

Las métricas por clase permiten identificar qué tipos de objetos son detectados con mayor o menor precisión.

Clases con mejor desempeño

Algunas clases presentan resultados relativamente altos:

Van

Precision: 0.541

Recall: 1.0

mAP50: 0.945

Esto indica que el modelo logra identificar correctamente esta clase en la mayoría de los casos.

Conventional Truck

Precision: 0.371

Recall: 0.75

mAP50: 0.801

Esto sugiere que el modelo logra reconocer adecuadamente este tipo de vehículo en las imágenes.

Grader

Precision: 1.0

mAP50: 0.995

Sin embargo, esta métrica puede estar influenciada por el bajo número de ejemplos en el dataset, lo cual puede inflar el resultado.

### Clases con bajo desempeño

Algunas clases presentan dificultades claras para el modelo.

Articulated Truck

mAP50: 0.015

mAP50-95: 0.012

Esto indica que el modelo prácticamente no logra detectar correctamente esta clase.

Bus

Precision: 0

Recall: 0

Esto sugiere que el modelo no detectó correctamente ningún bus en el conjunto de validación.

Person

Precision: 0.494

Recall: 0.185

Esto indica que el modelo tiene dificultades para detectar personas, posiblemente debido al tamaño pequeño del objeto en la imagen.

### Principales causas de error

Los resultados observados pueden explicarse por varios factores:

1. Dataset pequeño

El conjunto de validación contiene solo 8 imágenes, lo cual limita la capacidad de evaluar correctamente el desempeño del modelo.

Datasets pequeños también afectan la capacidad del modelo para aprender patrones visuales robustos.

### 2. Desbalance entre clases

Algunas clases tienen muy pocos ejemplos:

Por ejemplo:

Agricultural Tractor: 1 instancia
Grader: 1 instancia

Esto dificulta que el modelo aprenda características representativas para esas clases.

### 3. Similitud visual entre equipos

Muchos equipos de construcción tienen características visuales similares:

loaders vs excavators

trucks vs articulated trucks

Esto puede generar confusión entre clases.

### 4. Tamaño del objeto en la imagen

Objetos pequeños o parcialmente visibles pueden ser difíciles de detectar.

Esto explica el recall bajo, ya que el modelo tiende a omitir algunos objetos presentes.

### Recomendaciones para mejorar el modelo

Para mejorar el desempeño del modelo se recomiendan las siguientes acciones:

### Aumentar el tamaño del dataset

Incorporar más imágenes para cada clase permitiría mejorar la generalización del modelo.

### Balancear las clases

Mantener una distribución más uniforme entre clases ayudaría a evitar sesgos durante el entrenamiento.

### Aplicar técnicas de data augmentation

Técnicas como:

rotación

escalado

variación de iluminación

pueden mejorar la capacidad del modelo para generalizar.

### Incrementar las épocas de entrenamiento

Entrenar el modelo durante más épocas podría mejorar la convergencia del entrenamiento.

### Posibles mejoras (Resumen)

Para mejorar el desempeño del modelo se recomienda:

- aumentar el tamaño del dataset
- balancear mejor las clases
- incluir mayor variedad de ángulos
- incorporar diferentes condiciones de iluminación
- aumentar el número de épocas de entrenamiento

### Conclusión

El modelo muestra un desempeño moderado con una precisión aceptable pero un recall relativamente bajo, lo que indica que el modelo tiende a ser conservador al detectar objetos.

Las principales limitaciones del modelo están asociadas al tamaño reducido del dataset y al desbalance entre clases, lo cual es común en proyectos iniciales de visión computacional.

Con un dataset más grande y balanceado, es probable que el desempeño del modelo mejore significativamente.
---

# 8. Gobernanza y Uso Responsable

Este proyecto sigue principios básicos de **IA responsable**.

### Privacidad

El dataset utilizado no contiene información personal identificable.
El dataset contiene imagenes variadas de espacios de construccion.
El dataset contiene imagenes empresariales.

### Limitaciones del modelo

El modelo puede presentar menor precisión en situaciones como:

- iluminación deficiente
- oclusiones parciales
- objetos pequeños
- tipos de maquinaria no vistos durante el entrenamiento

### Riesgos

Los modelos de detección pueden generar:

- falsos positivos
- falsos negativos

Por lo tanto, el sistema no debe ser utilizado como único mecanismo de decisión en entornos críticos de seguridad.

---

# 9. Cómo Reproducir el Proyecto

Para reproducir el entrenamiento o inferencia del modelo:

1. Abrir el notebook del proyecto en **Google Colab**
2. Instalar las dependencias necesarias:

pip install ultralytics roboflow


3. Autenticarse en Roboflow
4. Descargar el dataset
5. Ejecutar el entrenamiento o cargar los pesos entrenados

---

# 10. Estructura del Repositorio
Tarea-M4T3_v2
│
├── notebooks
│
├── docs
│
├── results
│ ├── curves
│ └── evidence
│
└── weights


---

# 11. Pesos del Modelo

Los pesos del modelo entrenado se encuentran en:


weights/best.pt


---

# 12. Licencia

Este proyecto se distribuye bajo la licencia **MIT License**.

---

# Autor

**Ambar Figueroa Figari**  
MAIC – Proyecto de Visión Computacional

