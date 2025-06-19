# PROYECTO FINAL DATA SCIENCE III

Análisis de Sentimientos de Críticas de Películas en Español  
Este proyecto analiza críticas de películas en español utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y modelos de machine learning, con especial foco en una red neuronal para clasificar el sentimiento de las reseñas como **positivas** o **negativas**.

Objetivo del proyecto  
Aplicar técnicas de NLP y Deep Learning para:
- Clasificar comentarios como positivos o negativos
- Analizar la distribución de puntuaciones de usuarios
- Evaluar la polaridad y subjetividad de las opiniones
- Visualizar la relación entre géneros, películas y valoraciones

En un contexto donde la percepción del público influye en el éxito de una producción, este análisis permite detectar tendencias y patrones de satisfacción o descontento en los espectadores.

El conjunto de datos está formado por:
film_name: Título de la película. género: Género de la película (comedia, terror, acción, etc.) film_avg_rate: Nota media de la película (votos de todos los usuarios) review_rate: Nota que el usuario que hace la crítica pone a la película. review_title: Título de la crítica. review_text: Crítica de la película.

Herramientas y tecnologías  
- Python  
- Pandas, Seaborn, Matplotlib  
- spaCy, TextBlob  
- TensorFlow / Keras  
- WordCloud, Scikit-learn

Pipeline del proyecto  
1. **Carga y exploración del dataset**  
   - Limpieza y combinación de título + texto de reseña  
   - Análisis de distribución de puntuaciones  
   - Detección de géneros y películas más comentadas

2. **Preprocesamiento de texto**  
   - Lematización con spaCy  
   - Eliminación de stopwords, puntuación, símbolos y ruido  
   - Tokenización, limpieza personalizada de palabras irrelevantes

3. **Análisis de sentimientos**  
   - Cálculo de polaridad (TextBlob)  
   - Clasificación binaria: `polarity > 0 → positivo`  
   - Visualización de distribución de sentimientos

4. **Modelado con Red Neuronal**  
   Se implementó un modelo secuencial con:
   - Entrada: TF-IDF vectorizado  
   - Capa  con 12 neuronas  
   - Dropout del 40%  
   - Capa de salida softmax para clasificación binaria

Resultados del modelo  
- **Accuracy final**: **78%**
- Métricas (conjunto de test):  
  - *Precisión positiva*: 0.80  
  - *Precisión negativa*: 0.76  
  - *F1-score general*: 0.78

Comportamiento durante el entrenamiento  
- EarlyStopping aplicado para evitar sobreajuste  
- La pérdida de validación se estabilizó en la época 4  
- Buen aprendizaje progresivo con leve sobreajuste controlado

Matriz de Confusión  
El modelo mostró un buen desempeño en predicciones positivas (TP = 1058), aunque con errores en negativos mal clasificados (FN = 273), lo que indica cierto sesgo hacia opiniones favorables.

Como mejora futura, se podría aplicar un balanceo de clases o incorporar técnicas avanzadas como sobremuestreo, submuestreo o asignación de pesos, con el fin de mejorar el rendimiento del modelo en la clasificación de reseñas negativas.
