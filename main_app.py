from fastapi import FastAPI
import pandas as pd
import numpy as np
import random
from typing import Union
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


app = FastAPI()

#http://127.0.0.1:8000 

@app.get("/")
def index():
    return ("Buen dia, mi nombre Elio Padilla. SoyHenry")

@app.get("/genero/{ano}")
def genero(ano:int):  
    df_generos = pd.read_csv("func_genero.csv")
    var = ano
    df_generos_año = df_generos[df_generos["ano"] == var]
    grouped = df_generos_año.groupby('genres').size().reset_index(name='counts')
    sorted_grouped = grouped.sort_values(by='counts', ascending=False).head()
    dictionary = sorted_grouped.set_index('genres').to_dict(orient='index')
    return dictionary

@app.get("/juegos/{ano}")
def juegos(ano:int):
    df_titulos = pd.read_csv("func_juego.csv")
    var = ano
    lista_titulos_año = df_titulos[df_titulos["ano"] == var]["title"].unique().tolist()
    dic = {"Juegos":lista_titulos_año}
    return str(dic)

@app.get("/specs/{ano}")
def espec(ano:int):  
    df_espec = pd.read_csv("func_specs.csv")
    var = ano
    df_espec_año = df_espec[df_espec["ano"] == var]
    grouped = df_espec_año.groupby('specs').size().reset_index(name='counts')
    sorted_grouped = grouped.sort_values(by='counts', ascending=False).head(5)
    dictionary = sorted_grouped.set_index('specs').to_dict(orient='index')
    return dictionary

@app.get("/earlyaccess/{ano}")
def earlyacces(ano:int):
    df_early = pd.read_csv("func_early.csv")
    df_early_año = df_early[df_early["ano"] == ano]
    var = df_early_año[df_early_año["early_access"] == True]["early_access"].count()
    return int(var)

@app.get("/sentiment/{ano}")
def sentimiento(ano:int):  
    df_sentiment = pd.read_csv("func_sentiment.csv")
    var = ano
    df_sentiment_año = df_sentiment[df_sentiment["ano"] == var]
    grouped = df_sentiment_año.groupby('sentiment').size().reset_index(name='counts')
    sorted_grouped = grouped.sort_values(by='counts', ascending=False)
    dictionary = sorted_grouped.set_index('sentiment').to_dict(orient='index')
    return dictionary

@app.get("/metascore/{ano}")
def metascore(ano:int):  
    df_metascore = pd.read_csv("func_metascore.csv")
    var = ano
    df_metascore_año = df_metascore[df_metascore["ano"] == var]
    df_metascore_año = df_metascore_año.drop_duplicates(subset=['title', 'metascore'])
    Ordenado = df_metascore_año[["title","metascore"]] .sort_values('metascore', ascending = False).head()
    dictionary = Ordenado.set_index("title").to_dict()
    return dictionary

@app.get("/prediccion/{especificaciones}/{genero}/{lanzamiento}/{sentimiento}/")
def prediccion(especificaciones:str,genero:str,lanzamiento:str,sentimiento:str):
    # Procedemos a descargar datos en estructuras de listas.
    atributos = ["escala_specs","escala_genres","escala_early","escala_sentiment"]
    #Coef_efectivo = [-0.0479443,  -0.10295373,  2.29347288,  0.14412646]
    #Interc_efectivo = [7.671460041749117]
    #Coef_general = [-0.21829186, -1.02333733,  0.18669143,  0.22316687]
    #Interc_general = [29.844878941820042]
    Coef_general = [-16.27248769, -29.91685814,   0.3901166,   18.24283764]
    Interc_general = [14.277519448113623]
    Coef_efectivo = [-3.72837432, -3.20498395,  2.3159781 , 13.70138819]
    Interc_efectivo = [6.084275255404705]
    # Descargamos el listado de todas las especificaciones del dataset.
    #Escala_especificaciones = [('Game demo', 0), ('SteamVR Collectibles', 1), ('Includes Source SDK', 2), ('Steam Turn Notifications', 3), ('Commentary available', 4),
    #('Valve Anti-Cheat enabled', 5), ('In-App Purchases', 6), ('Local Co-op', 7), ('Captions available', 8), ('MMO', 9), ('Online Co-op', 10), ('Local Multi-Player', 11),
    #('Includes level editor', 12), ('Steam Workshop', 13), ('Cross-Platform Multiplayer', 14), ('Online Multi-Player', 15), ('Stats', 16), ('Shared/Split Screen', 17),
    #('Co-op', 18), ('Steam Leaderboards', 19), ('Partial Controller Support', 20), ('Downloadable Content', 21), ('Multi-player', 22), ('Full controller support', 23),
    #('Steam Cloud', 24), ('Steam Trading Cards', 25), ('Steam Achievements', 26), ('Single-player', 27)]
    Escala_especificaciones = [('Single-player', 0.21), ('Steam Achievements', 0.14), ('Steam Trading Cards', 0.11), ('Steam Cloud', 0.08), ('Full controller support', 0.06),
    ('Multi-player', 0.06), ('Downloadable Content', 0.05), ('Partial Controller Support', 0.05), ('Steam Leaderboards', 0.04), ('Co-op', 0.03),
    ('Shared/Split Screen', 0.03), ('Stats', 0.02), ('Online Multi-Player', 0.02), ('Cross-Platform Multiplayer', 0.02), ('Steam Workshop', 0.02),
    ('Includes level editor', 0.01), ('Local Multi-Player', 0.01), ('Online Co-op', 0.01), ('MMO', 0.01), ('Captions available', 0.01),
    ('Local Co-op', 0.01), ('In-App Purchases', 0.01), ('Valve Anti-Cheat enabled', 0.0), ('Commentary available', 0.0), ('Steam Turn Notifications', 0.0),
    ('Includes Source SDK', 0.0), ('SteamVR Collectibles', 0.0), ('Game demo', 0.0)]
    lista_spec = []
    lista_spec_esc = []
    for i in range(0,len(Escala_especificaciones)):
        lista_spec.append(Escala_especificaciones[i][0])
        lista_spec_esc.append(Escala_especificaciones[i][1])
    # Descargamos todos los valores de generos en el dataset.
    #Escala_generos = [('Photo Editing', 0), ('Audio Production', 1), ('Video Production', 2), ('Software Training', 3), ('Education', 4), ('Animation &amp; Modeling', 5),
    #('Utilities', 6), ('Web Publishing', 7), ('Design &amp; Illustration', 8), ('Free to Play', 9), ('Massively Multiplayer', 10), ('Racing', 11), ('Sports', 12),
    #('Early Access', 13), ('RPG', 14), ('Simulation', 15), ('Casual', 16), ('Strategy', 17), ('Adventure', 18), ('Action', 19), ('Indie', 20)]
    Escala_generos = [('Indie', 0.23), ('Action', 0.2), ('Adventure', 0.12), ('Strategy', 0.1), ('Casual', 0.09), ('Simulation', 0.08), ('RPG', 0.07),
    ('Early Access', 0.02), ('Sports', 0.02), ('Racing', 0.02), ('Massively Multiplayer', 0.02), ('Free to Play', 0.02), ('Design &amp; Illustration', 0.0),
    ('Web Publishing', 0.0), ('Utilities', 0.0), ('Animation &amp; Modeling', 0.0), ('Education', 0.0), ('Software Training', 0.0),
    ('Video Production', 0.0), ('Audio Production', 0.0), ('Photo Editing', 0.0)]
    lista_gen = []
    lista_gen_esc = []
    for i in range(0,len(Escala_generos)):
        lista_gen.append(Escala_generos[i][0])
        lista_gen_esc.append(Escala_generos[i][1])
    # Descargamos todos los valores de sentimiento en el dataset
    #Escala_sentimiento = [('Overwhelmingly Negative', 0), ('Very Negative', 1), ('Negative', 2), ('Overwhelmingly Positive', 3), ('9 user reviews', 4), ('8 user reviews', 5),
    #( '7 user reviews', 6), ('6 user reviews', 7), ('Mostly Negative', 8), ('5 user reviews', 9), ('4 user reviews', 10), ('3 user reviews', 11), ('2 user reviews', 12),
    #('1 user reviews', 13), ('Positive', 14), ('Mostly Positive', 15), ('Mixed', 16), ('Very Positive', 17)]
    Escala_sentimiento = [('Very Positive', 0.18), ('Mixed', 0.16), ('Mostly Positive', 0.13), ('Positive', 0.12), ('1 user reviews', 0.07), ('2 user reviews', 0.06),
    ('3 user reviews', 0.05), ('4 user reviews', 0.04), ('5 user reviews', 0.03), ('Mostly Negative', 0.03), ('6 user reviews', 0.03), ('7 user reviews', 0.03),
    ('8 user reviews', 0.02), ('9 user reviews', 0.02), ('Overwhelmingly Positive', 0.02), ('Negative', 0.0), ('Very Negative', 0.0), ('Overwhelmingly Negative', 0.0)]
    lista_sent = []
    lista_sent_esc = []
    for i in range(0,len(Escala_sentimiento)):
        lista_sent.append(Escala_sentimiento[i][0])
        lista_sent_esc.append(Escala_sentimiento[i][1])
    # Verificamos que el string de caracteres ingresado conincida con los valores del datset.
    if especificaciones in lista_spec:
        Exist_spec = True
        resp_usuario1 = True
    else:
        Exist_spec = False
        resp_usuario1 = False
    if genero in lista_gen:
        Exist_gen = True
        resp_usuario2 = True
    else:
        Exist_gen = False
        resp_usuario2 = False
    if sentimiento in lista_sent:
        Exist_sent = True
        resp_usuario3 = True
    else:
        Exist_sent = False
        resp_usuario3 = False
    if lanzamiento == "True" or lanzamiento == "1":
        Exist_lanz = True
        resp_usuario4 = True
    else:
        Exist_lanz = False
        resp_usuario4 = False
    # Si los valores ingrsados a la fucncion existen, se procede obtener el valor numerico asociado segun las escalas predefinidas para cada columna en el modelo ML
    if Exist_spec:
        var_spec = float(lista_spec_esc[lista_spec.index(especificaciones)])
    else:
        var_spec = float(-1)
    if Exist_gen:
        var_gen = float(lista_gen_esc[lista_gen.index(genero)])  #lista[lista.index(num)]
    else:
        var_gen = float(-1)
    if Exist_sent:
        var_sent = float(lista_sent_esc[lista_sent.index(sentimiento)])
    else:
        var_sent = float(-1)
    if Exist_lanz:
        var_lanz = float(1)
    else:
        var_lanz = float(0)
    # Si efectivmente los valores existen en el dataset. Se hace uso de sus valores escalados y se procede a calcular su proyeccion
    if resp_usuario1 and resp_usuario2 and resp_usuario3:
        variables = [var_spec,var_gen,var_lanz,var_sent]
        # Dado que se desarrollo el modelo de ML medante REGRESION LINEAL se hace uso de los coeficientes del plano a fin de calclar el valor.
        Precio_general = Interc_general[0] + (Coef_general[0] * variables[0]) + (Coef_general[1] * variables[1]) + (Coef_general[2] * variables[2]) + (Coef_general[3] * variables[3])
        Precio_efectivo = Interc_efectivo[0] + (Coef_efectivo[0] * variables[0]) + (Coef_efectivo[1] * variables[1]) + (Coef_efectivo[2] * variables[2]) + (Coef_efectivo[3] * variables[3])
        # Dado que la distribucion del precio titne particularidades en su comportamiento, se utiliza este a fin de estiar el precio proyectado segun dos probabilidades:
        # P(que e precio tenga una distribucion aprox uniforme entre 0 y 19,98 $) y la P(que el precio exceda hacia valores entre 19.99 hasta 771 $).
        # En funcion del resutado aleatorio la proyeccion de precio se emite con uno de los dos modelos de regresion predefinidos. Igualmente su variacion y precio mas probable.
        numero_aleatorio = random.randint(0, 100)
        if numero_aleatorio > 85:
            respuesta = Precio_general
            variacion = 13.44
            precio_probable = [abs(Precio_general-39.99),abs(Precio_general-19.99),abs(Precio_general-29.99)]
            minimo = min(precio_probable)
            if minimo == precio_probable[0]:
                p_probable = 39.99
            elif minimo == precio_probable[1]:
                p_probable = 19.99
            else: p_probable = 29.99 
        else:
            respuesta = Precio_efectivo
            variacion = 4.37
            precio_probable = [abs(Precio_efectivo-9.99),abs(Precio_efectivo-4.99),abs(Precio_efectivo-14.99)]
            minimo = min(precio_probable)
            if minimo == precio_probable[0]:
                p_probable = 9.99
            elif minimo == precio_probable[1]:
                p_probable = 4.99
            else: p_probable = 14.99 
            entregable = {'Precio_proyectado':round(respuesta,2),'Variacion': round(variacion,2), 'Precio_Probable':round(p_probable,2)}
            #diccionario = {'clave1': 'dato1', 'clave2': 'dato2', 'clave3': 'dato3'}
        return str(entregable)
    else:
        return "-1"