### Cargar paquetes
####### prueba
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler ## escalar variables 

####Este archivo contienen funciones utiles a utilizar en diferentes momentos del proyecto

def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]

    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer( strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)
    imputer_c.get_params()
    imputer_n.get_params()

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)


    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)
    df_c.info()
    df_n.info()

    df =pd.concat([df_n,df_c],axis=1)
    return df


def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= modelo.feature_names_in_[sel.get_support()]
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos



def preparar_datos(df):
    # Cargar listas y modelo
    list_cat = joblib.load("salidas\\list_cat.pkl")
    list_dummies = joblib.load("salidas\\list_dummies.pkl")
    var_names = joblib.load("salidas\\var_names.pkl")
    scaler = joblib.load("salidas\\scaler.pkl")

    # Seleccionar solo las columnas numéricas
    df_numeric = df.select_dtypes(include=['number'])

    # Imputar valores faltantes solo en las columnas numéricas
    imputer = SimpleImputer(strategy='median')
    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    # Mantener las columnas no numéricas sin cambios
    df_non_numeric = df.select_dtypes(exclude=['number'])

    # Concatenar las columnas numéricas imputadas con las columnas no numéricas originales
    df_processed = pd.concat([df_numeric_imputed, df_non_numeric], axis=1)

    # Crear variables dummy y escalar los datos
    df_dummies = pd.get_dummies(df_processed, columns=list_dummies)
    df_dummies = df_dummies.loc[:, ~df_dummies.columns.isin(['Attrition', 'EmployeeID', 'retirementDate', 'resignationReason'])]
    X_scaled = scaler.transform(df_dummies)
    X = pd.DataFrame(X_scaled, columns=df_dummies.columns)[var_names]
  
    return X