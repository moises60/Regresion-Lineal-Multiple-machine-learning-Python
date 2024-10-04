# Análisis de Regresión Lineal Múltiple en Python
![Regresión Lineal Múltiple](assets/Predicciones%20vs%20Valores%20Reales%20(Modelo%20Óptimo).png)
## Tabla de Contenidos

- [Descripción](#descripción)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Instalación](#instalación)
- [Uso](#uso)
- [Descripción del Dataset](#descripción-del-dataset)
- [Metodología](#metodología)
  - [1. Creación del Dataset](#1-creación-del-dataset)
  - [2. Preprocesamiento de Datos](#2-preprocesamiento-de-datos)
  - [3. Modelado Inicial](#3-modelado-inicial)
  - [4. Eliminación Hacia Atrás (Backward Elimination)](#4-eliminación-hacia-atrás-backward-elimination)
  - [5. Evaluación y Comparación de Modelos](#5-evaluación-y-comparación-de-modelos)


## Descripción

Este proyecto tiene como objetivo implementar y analizar un modelo de **Regresión Lineal Múltiple** utilizando Python 3.12 y Spyder 5. Se abordan todas las etapas del análisis de datos, desde la creación de un dataset realista hasta la evaluación y optimización del modelo mediante la técnica de **Eliminación Hacia Atrás (Backward Elimination)**. Además, se generan visualizaciones que ayudan a comprender mejor el comportamiento y rendimiento del modelo.

## Tecnologías Utilizadas

- **Lenguaje de Programación:** Python 3.12
- **Entorno de Desarrollo:** Spyder 5
- **Librerías Principales:**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `statsmodels`

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo y ejecutar el proyecto:

## Clonar el Repositorio
    git clone https://github.com/moises60/Regresion-Lineal-Multiple-machine-learning-Python.git

## Acceder a la Carpeta
    cd Regresion-Lineal-Multiple-machine-learning-Python

## Uso
El proyecto consta de dos scripts principales: 
- python crear_dataset_realista.py: Este script genera un dataset ficticio pero realista que simula salarios basados en diversas características de los empleados.
- python regresion_lineal_realista.py: Este script carga el dataset generado, realiza el preprocesamiento de datos, ajusta un modelo de regresión lineal múltiple inicial, optimiza el modelo mediante eliminación hacia atrás y finalmente evalúa y compara ambos modelos.

## Descripción del Dataset
El dataset simula información sobre empleados y sus respectivos salarios anuales. Las variables incluidas son:
- Edad: Edad del empleado (18 a 65 años).
- Experiencia: Años de experiencia laboral (0 a Edad-18).
- Educacion: Nivel educativo del empleado (Secundaria, Pregrado, Maestría, Doctorado).
- Departamento: Departamento al que pertenece el empleado (Ventas, Marketing, Finanzas, Tecnología, Recursos Humanos).
- Horas_Trabajo_Semana: Horas de trabajo por semana (30 a 50 horas).
- Salario: Salario anual en dólares (variable dependiente), calculado en función de las otras variables con un componente de error aleatorio.

## Metodología
## 1. Creación del Dataset
Se generó un dataset con 100 muestras donde cada observación representa a un empleado con características realistas. Las variables independientes incluyen edad, experiencia, nivel educativo, departamento y horas de trabajo semanales. El salario se calcula considerando:

- Salario Base por Educación: A mayor nivel educativo, mayor salario base.
- Incremento por Experiencia: Cada año adicional de experiencia aumenta el salario.
- Incremento por Departamento: Algunos departamentos ofrecen salarios más altos.
- Impacto de las Horas de Trabajo: Más horas trabajadas incrementan el salario.
- Error Aleatorio: Se añade un componente de ruido para simular variabilidad real.


## 2. Preprocesamiento de Datos
- Codificación de Variables Categóricas: Se utilizaron técnicas de OneHotEncoder para convertir variables categóricas (Educacion y Departamento) en variables numéricas, evitando la trampa de las variables ficticias mediante drop='first'.
- División del Dataset: Se dividió el dataset en conjuntos de entrenamiento (80%) y prueba (20%) para evaluar el rendimiento del modelo.

## 3. Modelado Inicial
Se ajustó un modelo de Regresión Lineal Múltiple utilizando todas las variables independientes. Este modelo sirve como punto de partida para la comparación posterior.
## Imágenes
![Regresión Lineal Múltiple](assets/Predicciones%20vs%20Valores%20Reales.png)

![Distribución de los Residuales (Modelo Inicial)](assets/Distribución%20de%20los%20Residuales%20(Modelo%20Inicial).png)

![Importancia de las Características en el Modelo Inicial](assets/Importancia%20de%20las%20Características%20en%20el%20Modelo%20Inicial.png)

![Comparación de Salarios Reales y Predichos (Modelo Inicial)](assets/Comparación%20de%20Salarios%20Reales%20y%20Predichos%20(Modelo%20Inicial).png)


## 4. Eliminación Hacia Atrás (Backward Elimination)
Se implementó la técnica de eliminación hacia atrás para optimizar el modelo:

Ajuste del Modelo OLS: Se ajusta un modelo de mínimos cuadrados ordinarios (OLS) usando statsmodels.
Evaluación de P-valores: Se identifican las variables con p-valores más altos.
Eliminación de Variables: Se elimina la variable con el p-valor más alto si excede un umbral de significancia (α = 0.05).
Repetición: El proceso se repite hasta que todas las variables restantes sean estadísticamente significativas.
![Predicciones vs Valores Reales (Modelo Óptimo)](assets/Predicciones%20vs%20Valores%20Reales%20(Modelo%20Óptimo).png)

![Distribución de los Residuales (Modelo Óptimo)](assets/Distribución%20de%20los%20Residuales%20(Modelo%20Óptimo).png)

![Importancia de las Características en el Modelo Óptimo](assets/Importancia%20de%20las%20Características%20en%20el%20Modelo%20Óptimo.png)

![Comparación de Salarios Reales y Predichos (Modelo Óptimo)](assets/Comparación%20de%20Salarios%20Reales%20y%20Predichos%20(Modelo%20Óptimo).png)

## 5. Evaluación y Comparación de Modelos
## Comparativa de Métricas
![Comparativa de Métricas entre Modelos](assets/Comparativa%20de%20Métricas%20entre%20Modelos.png)
| Modelo  | MAE      | MSE          | R² Score |
|---------|----------|--------------|----------|
| Inicial | 3,617.15 | 22,833,254.07 | 0.91     |
| Óptimo  | 3,532.28 | 21,815,496.12 | 0.91     |

El modelo óptimo presenta una ligera mejora en MAE y MSE sin perder capacidad explicativa, lo que lo hace más eficiente al utilizar menos variables.

