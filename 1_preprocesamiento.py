### Cargar paquetes 
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import Funciones as funciones  ###archivo de funciones propias
import sys ## saber ruta de la que carga paquetes
import seaborn as sns
import matplotlib.pyplot as plt

### Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanda agrega una ruta

### Cargar tablas de datos desde github ###

employees=("DATA/employee_survey_data.csv")  
general=("DATA/general_data.csv")  
manager=("DATA/manager_survey.csv")
retirement=("DATA/retirement_info.csv")

df_employees=pd.read_csv(employees)
df_general=pd.read_csv(general)
df_manager=pd.read_csv(manager)
df_retirement=pd.read_csv(retirement)

#Filtro de fecha de retiro a año 2016 
df_retirement['retirementDate'] = pd.to_datetime(df_retirement['retirementDate'])
df_retirement = df_retirement[df_retirement['retirementDate'].dt.year == 2016 ]  
df_retirement['retirementDate'].value_counts()
### Verificar lectura correcta de los datos
df_employees.sort_values(by=['EmployeeID'],ascending=1).head(100)
df_general.sort_values(by=['EmployeeID'],ascending=0).head(5)
df_manager.sort_values(by=['EmployeeID'],ascending=0).head(100)
df_retirement.sort_values(by=['EmployeeID'],ascending=0).head(100)

### Resumen con información tablas faltantes y tipos de variables y hacer correcciones
df_general.info(verbose=True)
df_employees.info()
df_manager.info()
df_retirement.info()

### Eliminar duplicados basados en la columna 'EmployeeID'
df_employees = df_employees.drop_duplicates(subset='EmployeeID', keep='first')
df_general = df_general.drop_duplicates(subset='EmployeeID', keep='first')
df_manager = df_manager.drop_duplicates(subset='EmployeeID', keep='first')
df_retirement = df_retirement.drop_duplicates(subset='EmployeeID', keep='first')

### Eliminar la columna sin nombre 
df_employees = df_employees.drop(columns=['Unnamed: 0'])
df_general = df_general.drop(columns=['Unnamed: 0'])
df_manager = df_manager.drop(columns=['Unnamed: 0'])
df_retirement = df_retirement.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])

### Verifica las columnas actuales
print(df_employees.columns)
print(df_general.columns)
print(df_manager.columns)
print(df_retirement.columns)

### Eliminar las variables con mismo valor
df_employees = df_employees.drop(columns=['DateSurvey'])
df_general = df_general.drop(columns=['EmployeeCount','Over18','StandardHours','InfoDate'])
df_manager = df_manager.drop(columns=['SurveyDate'])
df_retirement = df_retirement.drop(columns=['retirementType'])

### Verifica las columnas actuales
print(df_employees.columns)
print(df_general.columns)
print(df_manager.columns)
print(df_retirement.columns)

### Reemplazar valores nulos por 0
df_general['NumCompaniesWorked'].fillna(0, inplace=True)
df_general['TotalWorkingYears'].fillna(0, inplace=True)
df_employees['EnvironmentSatisfaction'].fillna(0, inplace=True)
df_employees['JobSatisfaction'].fillna(0, inplace=True)
df_employees['WorkLifeBalance'].fillna(0, inplace=True)


### Verificar el resultado
df_general.info(verbose=True)
df_employees.info()
df_manager.info()
df_retirement.info()

### Combinar los primeros tres DataFrames
df_combined = pd.merge(df_general, df_employees, on='EmployeeID', how='inner')
df_combined = pd.merge(df_combined, df_manager, on='EmployeeID', how='inner')
### Combinar con el DataFrame de retiro
df_final = pd.merge(df_combined, df_retirement, on='EmployeeID', how='left')
df_final

### cambiar los valores nulos de la columna 'retirementDate' por 'NoRetirement'
df_final['retirementDate'].fillna(0, inplace=True)
### cambiar los valores nulos de la columna 'resignationReason' por 'NoReason'
df_final['resignationReason'].fillna('NoRetirement', inplace=True)

df_final.value_counts('resignationReason')

### cambair la variable Attrition a binomial 0 y 1
df_final['Attrition'] = df_final['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df_final

# cambiar el tipo de variable IDEmpleado 
df_final['EmployeeID'] = df_final['EmployeeID'].astype(str)

df_final.info()
df_final.describe

df_final.to_csv('DATA/df_final.csv', index=False)







