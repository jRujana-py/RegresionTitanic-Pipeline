import pandas as pd
from sklearn.impute import SimpleImputer   
from sklearn.preprocessing import OneHotEncoder   #OneHotEncoder para una columna por cada variable categórica
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Leemos los datos
datos = pd.read_csv("G:/Python/Datasets Python/Precios de casas/EntrenoPreciosCasas.csv")
prueba = pd.read_csv("G:/Python/Datasets Python/Precios de casas/PruebaPreciosCasas.csv")

xPrueba = prueba.drop(["Id"], axis=1)
ID = prueba["Id"]

#Definimos x y y
x = datos.drop(['SalePrice', "Id"], axis=1)
y = datos.SalePrice

#Identificamos las columnas categóricas y las numéricas
colCat = [col for col in x.columns if x[col].dtype == 'object']

colNum = [col for col in x.columns if x[col].dtype in ['int64', 'float64']]

#Definimos como vamos a imputar valores numéricos perdidos
numTrans = SimpleImputer(strategy='constant')

#Definimos como vamos a imputar valores categóricos y transformarlos a valores numéricos
catTrans = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

#Definimos el preprocesador para valors numéricos y categóricos
preprocesador = ColumnTransformer(
    transformers=[
        ('num', numTrans, colNum),
        ('cat', catTrans, colCat)
    ])


#Definimos el modelo que vayamos a utilizar para realizar las predicciones
reg = LinearRegression()

#Definimos el pipeline
pipeline = Pipeline(steps=[
        ('preprocessor', preprocesador),
        ('model', reg)
    ])

#Entrenamos a el pipeline
pipeline.fit(x, y)

#Realizamos predicciones
preds = pipeline.predict(xPrueba)

#df = pd.DataFrame({"Id": ID, "SalePrice": preds})
#df.to_csv("G:/Python/Datasets Python/Precios de casas/SubmissionHousePrice.csv", index=False)