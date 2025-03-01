# se importa la librería de Pandas
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib


def read_and_clean_data():
    df = pd.read_csv('libertadores-results-ds.csv')
    # se eliminan los resultados donde la columna Round es igual a Qualifying Match
    df = df[df['Round'] != 'Qualifying Match']
    # se eliminan las colunas de Edition y Date
    df = df.drop(['Edition', 'Date'], axis=1)
    # se buscan los registros que en el campo Round comienzan y se les asigna el valor
    df.loc[df['Round'].str.contains('Group'), 'Round'] = 'Groups'

    # Se crea una nueva columna llamada 'Score' con valores iniciales de 0
    df['Score'] = 0

    # Se utiliza la función .loc para asignar valores a la columna 'Score' según las condiciones
    df.loc[df['Home Score'] > df['AwayScore'], 'Score'] = 1
    df.loc[df['Home Score'] < df['AwayScore'], 'Score'] = -1
    # se eliminan los campos Home Score y	AwayScore
    df = df.drop(['Home Score', 'AwayScore'], axis=1)
    return df


def train_model(df):
    # Separar variables predictoras y objetivo
    X = df[['Round', 'Home Club', 'Away Club']]
    y = df['Score']
    # Preprocesamiento: codificación de variables categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Round', 'Home Club', 'Away Club'])
        ]
    )
    # Definición del pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
    ])
    # División de los datos en entrenamiento y prueba (aunque en este ejemplo la data es muy pequeña)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Entrenamiento del modelo
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, filename):
    joblib.dump(model, filename)


def train():
    df = read_and_clean_data()
    model = train_model(df)
    save_model(model, 'pipeline_total.pkl')
    save_model(model, 'mejor_modelo.pkl')
