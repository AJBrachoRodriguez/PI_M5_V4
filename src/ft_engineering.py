# libreías
# librerías
import pandas as pd
from carga_datos import cargarDatos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# aquí cargamos los datos
df = cargarDatos()

# luego, hacemos un preview de los datos para entender su estructura
print(df.head())
print(df.columns)
print(df.describe())

# Paso 1: identificar las features y el target
X = df.drop("Pago_atiempo", axis=1) # features
y = df["Pago_atiempo"] # target

# Paso 2: identificar columnas numéricas y categóricas
num_features = X. select_dtypes('number').columns # variables numéricas
cat_features = X.select_dtypes('object').columns # variables categóricas

print("Features numéricas:", num_features)
print("Features categóricas:", cat_features)

# Paso 3: crear pipelines para cada tipo de feature

## Paso 3.1: pipeline para features numéricas
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ]
)

## Paso 4:Combinar los transformadores en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features)
    ]
)