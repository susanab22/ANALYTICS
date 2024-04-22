### Cargar paquetes
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


###Ruta directorio qué tiene paquetes
sys.path
sys.path.append('c:\\cod\\LEA3_HR\\data') ## este comanoa agrega una ruta
df_final=("DATA/df_final.csv")  
df_final=pd.read_csv(df_final)
#df_final['EmployeeID'] = df_final['EmployeeID'].astype(str)
df_final.columns

### explorar variable respuesta ###
fig=df_final.Attrition.hist(bins=50,ec='black') ## no hay atípicos
fig.grid(False)
plt.show()

df_final.shape


#Correción del tipo de cada variable

df_final.info()
df_final.value_counts('NumCompaniesWorked')


df_final['EnvironmentSatisfaction'] = df_final['EnvironmentSatisfaction'].astype('category')
df_final['JobSatisfaction'] = df_final['JobSatisfaction'].astype('category')
df_final['WorkLifeBalance'] = df_final['WorkLifeBalance'].astype('category')
df_final['Education'] = df_final['Education'].astype('category')
df_final['JobInvolvement'] = df_final['JobInvolvement'].astype('category')
df_final['PerformanceRating'] = df_final['PerformanceRating'].astype('category')
df_final['JobLevel'] = df_final['JobLevel'].astype('category')
df_final['resignationReason'] = df_final['resignationReason'].astype('category')


df_final['JobLevel'] = df_final['JobLevel'].astype('category')
df_final['StockOptionLevel'] = df_final['StockOptionLevel'].astype('category')
df_final['TrainingTimesLastYear'] = df_final['TrainingTimesLastYear'].astype('category')
df_final['NumCompaniesWorked'] = df_final['NumCompaniesWorked'].astype('int64')

df_final['retirementDate'] = pd.to_datetime(df_final['retirementDate'], errors='coerce')

df_final['Attrition'] = df_final['Attrition'].astype('category')
df_final['BusinessTravel'] = df_final['BusinessTravel'].astype('category')
df_final['Department'] = df_final['Department'].astype('category')
df_final['EducationField'] = df_final['EducationField'].astype('category')
df_final['Gender'] = df_final['Gender'].astype('category')
df_final['JobRole'] = df_final['JobRole'].astype('category')
df_final['MaritalStatus'] = df_final['MaritalStatus'].astype('category')

df_final.info()


### explorar variables numéricas  ###
df_final.hist(figsize=(15, 15), bins=20)
plt.suptitle('Distribución de Variables Numéricas', y=1.02)
plt.show()

# Crear grafica de retiros por mes
df_final.set_index('retirementDate').resample('M').size().plot()
plt.title('Retiros por mes')
plt.xlabel('FechaDeRetiro')
plt.ylabel('Empleados retirados')
plt.show()

### explorar variables categóricas
# Configurar estilo de Seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))

# Lista de variables categóricas
categories = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'resignationReason']

# Iterar sobre las variables y crear gráficas en un solo gráfico
for i, column in enumerate(categories, start=1):
    plt.subplot(2, 4, i)
    sns.countplot(x=column, data=df_final, palette='Set3')  # Puedes ajustar la paleta según tus preferencias
    plt.title(f'Distribution of {column}', fontsize=14)
    plt.xlabel("")  # Eliminar la etiqueta del eje x para mejorar el diseño
    plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x para mejorar la legibilidad

plt.suptitle('Exploration of Categorical Variables', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

### Variable respuesta VS categoricas  ###   
variables_analisis = [
    'EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
    'BusinessTravel', 'Department', 'Education', 'EducationField', 'Gender',
    'JobLevel', 'JobRole', 'MaritalStatus', 'StockOptionLevel',
    'TrainingTimesLastYear', 'JobInvolvement', 'PerformanceRating'
]
sns.set(style="whitegrid")
fig, axs = plt.subplots(3, 5, figsize=(14, 8))
fig.suptitle('Análisis de Attrition', fontsize=16)

for var, ax in zip(variables_analisis, axs.flatten()):
    sns.countplot(x=var, hue='Attrition', data=df_final, ax=ax, palette='viridis')
    ax.set_title(var, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.subplots_adjust(wspace=0.25, hspace=1.5)

plt.legend(title='Attrition', bbox_to_anchor=(1.05, 2), loc='upper left')
plt.show()

### Seleccionar solo las columnas categóricas
df_categoricas = df_final.select_dtypes(include=['category'])

### Crear una matriz vacía para almacenar los valores de p-valor
p_values = []

# Correlación de chi-cuadrado y los p-valores
for col1 in df_categoricas.columns:
    row_p_values = []
    for col2 in df_categoricas.columns:
        if col1 == col2:
            row_p_values.append(1.0)  # Poner 1.0 en la diagonal principal
        else:
            contingency_table = pd.crosstab(df_categoricas[col1], df_categoricas[col2])
            _, p, _, _ = chi2_contingency(contingency_table)
            row_p_values.append(p)
    p_values.append(row_p_values)

# Crear un DataFrame de p-valores
p_value_df = pd.DataFrame(p_values, columns=df_categoricas.columns, index=df_categoricas.columns)

plt.figure(figsize=(10, 8))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(p_value_df, annot=True, cmap=cmap, fmt=".2f")
plt.title("Matriz de Asociación (p-Valores) entre Variables Categóricas")
plt.show()

### Attrition (Categorica) vs numericas ###
# Crear una figura con tres filas y cinco columnas
fig, axs = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle('Attrition (Categorica) vs Variables Numericas', fontsize=16)

# Colores para Attrition (0: 'blue', 1: 'orange')
colors = {0: 'blue', 1: 'orange'}

# Crear los gráficos
for i, variable in enumerate(
    [
        'Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
        'PercentSalaryHike', 'YearsAtCompany', 'YearsSinceLastPromotion',
        'YearsWithCurrManager', 'TotalWorkingYears',
    ]
):
    # Crear un gráfico de dispersión para la variable numérica
    sns.scatterplot(x=df_final[variable], y=df_final['Attrition'], ax=axs[i // 5, i % 5], alpha=0.5, s=30, hue=df_final['Attrition'], palette=colors)
    
    # Ajustar etiquetas y título de la gráfica
    axs[i // 5, i % 5].set_title(variable, fontsize=12)
    axs[i // 5, i % 5].set_xlabel(variable)
    axs[i // 5, i % 5].set_ylabel('Attrition')
    axs[i // 5, i % 5].legend().set_visible(False)

# Ajustar el diseño de la figura
plt.subplots_adjust(wspace=0.5, hspace=0.6)

# Mostrar la figura
plt.show()

### Relación entre numericas  ###

df_numericas = df_final.select_dtypes(include=[np.number])
correlation_matrix = df_numericas.corr()
mask = np.triu(correlation_matrix, k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu", mask=mask)
plt.title("Matriz de Correlación de Variables Numéricas")
plt.show()
#-----------------------------------------------------------------

### Selección de variables ###
df_final_V2 = df_final.copy()

# convertir las variables categoricas en dummies
df_final_V2 = pd.get_dummies(df_final_V2, columns=['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'EducationField'])

# cambiar el tipo de las variables dummies a int
for column in ['BusinessTravel', 'Department', 'Gender', 'JobRole', 'MaritalStatus', 'EducationField']:
    for dummy_variable in df_final_V2.columns:
        if dummy_variable.startswith(column):
            df_final_V2[dummy_variable] = df_final_V2[dummy_variable].astype(int)

# Print the first 3 rows
df_final_V2.head()

### Seleccion de variables por metodo Wrapper ###
#Backward selection
df_final_V2_int = df_final_V2.select_dtypes(include = ["number"]) # filtrar solo variables númericas
#df_final_V2_int = df_final_V2_int.drop(['Attrition', 'retirementDate'], axis = 1) # excluir 'Attrition' y 'retirementDate'
y = df_final_V2['Attrition']
df_final_V2_int.head()

# Normalización de variables categoricas ordinales
from sklearn.preprocessing import MinMaxScaler
df_final_V2_norm = df_final_V2_int.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler()  # asignar el tipo de normalización
sv = scaler.fit_transform(df_final_V2_norm.iloc[:, :])  # normalizar los datos
df_final_V2_norm.iloc[:, :] = sv  # asignar los nuevos datos
df_final_V2_norm.head()

from sklearn.feature_selection import RFE
# Función recursiva de selección de características
def recursive_feature_selection(X,y,model,k): #model=modelo que me va a servir de estimador para seleccionar las variables
                                              # K = variables que se quiere tener al final
  rfe = RFE(model, n_features_to_select=k, step=1)# step=1 cada cuanto el toma la sicesion de tomar una caracteristica; paso de analisis de caracteristicas
  fit = rfe.fit(X, y)
  c2_var = fit.support_
  print("Num Features: %s" % (fit.n_features_))
  print("Selected Features: %s" % (fit.support_))
  print("Feature Ranking: %s" % (fit.ranking_))

  return c2_var # estimador de las variables seleccioandas

# Establecer Estimador
model = LinearRegression() # algoritmo a travez del cual se van a encontrar las variables

# Obtener columnas seleciconadas - (3 caracteristicas)
df_final_V2_var = recursive_feature_selection(df_final_V2_norm, y, model,7) # x_int =  conjunto caracteristicas numericas
                                                        # y = variable respueta
                                                        # model = modelo que se definio para estimar variables
                                                        # k = numero de variables que se quiere al final

# Nuevo conjunto de datos
df_final_V3 = df_final_V2_int.iloc[:,df_final_V2_var]
df_final_V3.head()


