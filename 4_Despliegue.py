import Funciones as funciones  ###archivo de funciones propias
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
# Cargar datos desde el archivo CSV
df_final = "DATA/df_final.csv"
df = pd.read_csv(df_final)


df_t= funciones.preparar_datos(df)



# Cargar modelo entrenado
rf_final = joblib.load("salidas\\rf_final.pkl")

# Realizar predicciones
predicciones = rf_final.predict(df_t)
pd_pred = pd.DataFrame(predicciones, columns=['Atrition'])

# Crear DataFrame con predicciones
perf_pred = pd.concat([df['EmployeeID'], df_t, pd_pred], axis=1)

# Guardar predicciones en archivos
perf_pred[['EmployeeID', 'Atrition']].to_excel("salidas\\prediccion.xlsx")
#Guardar importancia de las caracteristicas a la hora de predecir
feature_names = df_t.columns
importances = pd.DataFrame({'Feature': feature_names, 'Importance': rf_final.feature_importances_})
importances.to_excel("salidas\\importances.xlsx")


# Ver las 10 predicciones más bajas
emp_pred_bajo = perf_pred.sort_values(by=["Atrition"], ascending=True).head(10)
emp_pred_bajo.set_index('EmployeeID', inplace=True)
pred = emp_pred_bajo.T
print(pred)