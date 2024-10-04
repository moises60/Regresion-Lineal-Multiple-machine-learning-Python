# -*- coding: utf-8 -*-
"""
Creado el Martes, 1 de Octubre de 2024

Este script genera un dataset realista para regresión lineal múltiple y lo guarda como 'datos_regresion_realista.csv'.
"""

import pandas as pd
import numpy as np

# Configuración de la semilla para reproducibilidad
np.random.seed(0)

# Crear datos
num_muestras = 100

# Variables independientes
edades = np.random.normal(35, 10, num_muestras).astype(int)  # Edad de los empleados
edades = np.clip(edades, 18, 65)  # Limitar edades entre 18 y 65 años

experiencias = edades - np.random.randint(18, 25, num_muestras)  # Años de experiencia
experiencias = np.clip(experiencias, 0, None)  # Experiencia no negativa

departamentos = np.random.choice(['Ventas', 'Marketing', 'Finanzas', 'Tecnología', 'Recursos Humanos'], num_muestras)

educacion_niveles = ['Secundaria', 'Pregrado', 'Grado', 'Doctorado']
educacion = np.random.choice(educacion_niveles, num_muestras, p=[0.2, 0.5, 0.25, 0.05])

horas_trabajo = np.random.normal(40, 5, num_muestras).astype(int)
horas_trabajo = np.clip(horas_trabajo, 30, 50)

# Salario base según nivel educativo
salario_base_educacion = {'Secundaria': 20000, 'Pregrado': 30000, 'Grado': 40000, 'Doctorado': 50000}

# Incremento salarial por experiencia (por año)
incremento_experiencia = 1000

# Incremento salarial por departamento
incremento_departamento = {
    'Ventas': 5000,
    'Marketing': 6000,
    'Finanzas': 7000,
    'Tecnología': 8000,
    'Recursos Humanos': 4000
}

# Cálculo del salario
salarios = []
for i in range(num_muestras):
    salario = salario_base_educacion[educacion[i]]
    salario += experiencias[i] * incremento_experiencia
    salario += incremento_departamento[departamentos[i]]
    salario += horas_trabajo[i] * 50  # Suponiendo que por cada hora de trabajo semanal se suma una cantidad al salario
    salario += np.random.normal(0, 5000)  # Error aleatorio
    salarios.append(salario)

# Crear DataFrame
datos = pd.DataFrame({
    'Edad': edades,
    'Experiencia': experiencias,
    'Educacion': educacion,
    'Departamento': departamentos,
    'Horas_Trabajo_Semana': horas_trabajo,
    'Salario': salarios
})

# Guardar a CSV
datos.to_csv('datos_regresion_realista.csv', index=False)

print("Dataset 'datos_regresion_realista.csv' creado exitosamente.")



