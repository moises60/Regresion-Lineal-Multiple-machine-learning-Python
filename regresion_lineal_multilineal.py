# -*- coding: utf-8 -*-
"""
Creado el Martes, 1 de Octubre de 2024

Este script realiza una regresión lineal múltiple sobre el dataset 'datos_regresion_realista.csv'.
Incluye visualizaciones para demostrar las ventajas de la regresión lineal múltiple y
una comparativa entre el modelo inicial y el modelo óptimo obtenido mediante Eliminación hacia atrás.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo de los gráficos
sns.set(style="whitegrid")

# Importar el dataset
datos = pd.read_csv('datos_regresion_realista.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(datos.head())

# Separar variables independientes (X) y dependiente (y)
X = datos[['Edad', 'Experiencia', 'Educacion', 'Departamento', 'Horas_Trabajo_Semana']]
y = datos['Salario']

# Codificar variables categóricas
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definir el transformador para las columnas categóricas
transformador = ColumnTransformer(
    transformers=[
        ('encoder_educacion', OneHotEncoder(drop='first'), ['Educacion']),
        ('encoder_departamento', OneHotEncoder(drop='first'), ['Departamento'])
    ],
    remainder='passthrough'  # Mantener las demás columnas sin transformar
)

X = transformador.fit_transform(X)

# Nombres de las características después de la codificación
nombres_educacion = transformador.named_transformers_['encoder_educacion'].get_feature_names_out(['Educacion'])
nombres_departamento = transformador.named_transformers_['encoder_departamento'].get_feature_names_out(['Departamento'])
caracteristicas = np.concatenate((nombres_educacion, nombres_departamento, ['Edad', 'Experiencia', 'Horas_Trabajo_Semana']))

# Convertir a DataFrame para facilitar el manejo
X = pd.DataFrame(X, columns=caracteristicas)

# Dividir el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Ajustar el modelo de Regresión Lineal Múltiple (Modelo Inicial)
from sklearn.linear_model import LinearRegression

modelo_inicial = LinearRegression()
modelo_inicial.fit(X_entrenamiento, y_entrenamiento)

# Realizar predicciones del modelo inicial
y_pred_inicial = modelo_inicial.predict(X_prueba)

# Evaluar el modelo inicial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_inicial = mean_absolute_error(y_prueba, y_pred_inicial)
mse_inicial = mean_squared_error(y_prueba, y_pred_inicial)
r2_inicial = r2_score(y_prueba, y_pred_inicial)

print("\nEvaluación del Modelo Inicial:")
print(f"Error Absoluto Medio (MAE): {mae_inicial:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse_inicial:.2f}")
print(f"R^2 Score: {r2_inicial:.2f}")

# Visualización de los resultados del modelo inicial

# 1. Gráfico de Predicciones vs Valores Reales (Modelo Inicial)
plt.figure(figsize=(10,6))
plt.scatter(y_prueba, y_pred_inicial, color='blue', label='Modelo Inicial')
plt.plot([y_prueba.min(), y_prueba.max()], [y_prueba.min(), y_prueba.max()], color='red', linewidth=2)
plt.title('Predicciones vs Valores Reales (Modelo Inicial)')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.legend()
plt.show()

# 2. Residuales del Modelo Inicial
residuals_inicial = y_prueba - y_pred_inicial

plt.figure(figsize=(10,6))
sns.histplot(residuals_inicial, kde=True, color='green')
plt.title('Distribución de los Residuales (Modelo Inicial)')
plt.xlabel('Residuales')
plt.show()

# 3. Importancia de las Características (Modelo Inicial)
importancia_inicial = modelo_inicial.coef_

plt.figure(figsize=(10,6))
sns.barplot(x=importancia_inicial, y=caracteristicas, palette='viridis', dodge=False)
plt.title('Importancia de las Características en el Modelo Inicial')
plt.xlabel('Coeficientes')
plt.ylabel('Características')
plt.legend([],[], frameon=False)  # Eliminar la leyenda
plt.show()

# 4. Gráfico de Comparación de Salarios Reales y Predichos (Modelo Inicial)
datos_resultado_inicial = pd.DataFrame({
    'Salario Real': y_prueba,
    'Salario Predicho': y_pred_inicial
}).reset_index(drop=True)

datos_resultado_inicial.plot(kind='bar', figsize=(12,6))
plt.title('Comparación de Salarios Reales y Predichos (Modelo Inicial)')
plt.xlabel('Muestra')
plt.ylabel('Salario')
plt.show()

## Construir el modelo óptimo de Regresión Lineal Múltiple utilizando la Eliminación hacia atrás
import statsmodels.api as sm

def backward_elimination(X, y, sl=0.05):
    variables = list(X.columns)
    while True:
        X_with_constant = sm.add_constant(X[variables])
        modelo = sm.OLS(y, X_with_constant).fit()
        p_values = modelo.pvalues.iloc[1:]  # Excluir el intercepto

        max_p_value = p_values.max()
        if max_p_value > sl:
            excluded_var = p_values.idxmax()
            print(f"Eliminando la variable: {excluded_var} con p-value = {max_p_value:.4f}")
            variables.remove(excluded_var)
        else:
            break
    return X[variables], modelo

print("\nIniciando la Eliminación hacia atrás...")
X_optimos, modelo_optimo = backward_elimination(X, y)

# Mostrar el resumen del modelo óptimo
print("\nResumen del Modelo Óptimo (OLS):")
print(modelo_optimo.summary())

# Dividir los datos en entrenamiento y prueba para el modelo óptimo
X_entrenamiento_opt, X_prueba_opt, y_entrenamiento_opt, y_prueba_opt = train_test_split(
    X_optimos, y, test_size=0.2, random_state=0
)

# Ajustar el modelo óptimo utilizando scikit-learn
modelo_optimo_sk = LinearRegression()
modelo_optimo_sk.fit(X_entrenamiento_opt, y_entrenamiento_opt)

# Realizar predicciones del modelo óptimo
y_pred_optimo = modelo_optimo_sk.predict(X_prueba_opt)

# Evaluar el modelo óptimo
mae_optimo = mean_absolute_error(y_prueba_opt, y_pred_optimo)
mse_optimo = mean_squared_error(y_prueba_opt, y_pred_optimo)
r2_optimo = r2_score(y_prueba_opt, y_pred_optimo)

print("\nEvaluación del Modelo Óptimo:")
print(f"Error Absoluto Medio (MAE): {mae_optimo:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse_optimo:.2f}")
print(f"R^2 Score: {r2_optimo:.2f}")

# Visualización de los resultados del modelo óptimo

# 1. Gráfico de Predicciones vs Valores Reales (Modelo Óptimo)
plt.figure(figsize=(10,6))
plt.scatter(y_prueba_opt, y_pred_optimo, color='blue', label='Modelo Óptimo')
plt.plot([y_prueba_opt.min(), y_prueba_opt.max()], [y_prueba_opt.min(), y_prueba_opt.max()], color='red', linewidth=2)
plt.title('Predicciones vs Valores Reales (Modelo Óptimo)')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.legend()
plt.show()

# 2. Residuales del Modelo Óptimo
residuals_optimo = y_prueba_opt - y_pred_optimo

plt.figure(figsize=(10,6))
sns.histplot(residuals_optimo, kde=True, color='green')
plt.title('Distribución de los Residuales (Modelo Óptimo)')
plt.xlabel('Residuales')
plt.show()

# 3. Importancia de las Características (Modelo Óptimo)
importancia_optimo = modelo_optimo_sk.coef_
caracteristicas_optimo = X_optimos.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importancia_optimo, y=caracteristicas_optimo, palette='viridis', dodge=False)
plt.title('Importancia de las Características en el Modelo Óptimo')
plt.xlabel('Coeficientes')
plt.ylabel('Características')
plt.legend([],[], frameon=False)  # Eliminar la leyenda
plt.show()

# 4. Gráfico de Comparación de Salarios Reales y Predichos (Modelo Óptimo)
datos_resultado_optimo = pd.DataFrame({
    'Salario Real': y_prueba_opt,
    'Salario Predicho': y_pred_optimo
}).reset_index(drop=True)

datos_resultado_optimo.plot(kind='bar', figsize=(12,6))
plt.title('Comparación de Salarios Reales y Predichos (Modelo Óptimo)')
plt.xlabel('Muestra')
plt.ylabel('Salario')
plt.show()

# Comparativa Exhaustiva entre el Modelo Inicial y el Modelo Óptimo

# Crear un DataFrame para comparar las métricas
comparativa = pd.DataFrame({
    'Modelo': ['Inicial', 'Óptimo'],
    'MAE': [mae_inicial, mae_optimo],
    'MSE': [mse_inicial, mse_optimo],
    'R^2 Score': [r2_inicial, r2_optimo]
})

print("\nComparativa de Métricas entre Modelo Inicial y Modelo Óptimo:")
print(comparativa)

# Visualización de la Comparativa de Métricas
comparativa_melted = comparativa.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor')

plt.figure(figsize=(10,6))
sns.barplot(x='Métrica', y='Valor', hue='Modelo', data=comparativa_melted, palette='Set2')
plt.title('Comparativa de Métricas entre Modelos')
plt.ylabel('Valor')
plt.show()
