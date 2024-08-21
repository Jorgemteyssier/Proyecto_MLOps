from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.title = "MLOps proyecto 1 recomendacion películas"

#Cargo el Dataset y lo convierto en un df
df_api=pd.read_parquet('Data_Api.parquet')

#1. cantidad_filmaciones_mes( Mes ): Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en el mes consultado en la totalidad del dataset.
#Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    
    #Creo un diccionario con los meses y el número que le corresponde
    meses_numero = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,"julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12 }
    
    #Paso a minusculas para evitar discrepancias tipo Junio != junio
    mes=mes.lower()
    #Validacion de entrada
    if mes not in meses_numero:
        return f'Error: Mes invalido. Intentalo de nuevo con un mes en español. Checa la ortografía'
    #Cambio el mes a numero 
    mes_num=meses_numero[mes]
    
    #Filtrado y conteo del mes en el df
    total=df_api[df_api['release_date'].dt.month==mes_num].shape[0]
    return f'{total} películas fueron estrenadas en el mes de {mes}'


#2.def cantidad_filmaciones_dia( Dia ): Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron estrenadas en día consultado en la totalidad del dataset.
#Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X

@app.get('/cantidad_filmaciones_dia/{dia}')

def cantidad_filmaciones_dia(dia: str):

    #Creo diccionario con los dias y sus números
    dia_numero={"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6}

    #Paso a minusculas
    dia=dia.lower()
    #Validación de entrada
    if dia not in dia_numero:
        return f'Error: Día invalido. Intentalo de nuevo con un día en español. Checa la ortografía'
    #Cambio a numero
    dia_num=dia_numero[dia]

    #Filtrado y conteo 
    total=df_api[df_api['release_date'].dt.dayofweek==dia_num].shape[0]
    return f'{total} películas fueron estrenadas en el día {dia}'

#3.def score_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, el año de estreno y el score.
#Ejemplo de retorno: La película X fue estrenada en el año X con un score/popularidad de X

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):

    #Para evitar errores de ortografia lo estandarizamos
    titulo = titulo.lower()

    #Buscamos el titulo en el df y tambien lo pasamos a minusculas
    movie=df_api[df_api['title'].str.lower()==titulo]


    #Validacion
    if movie.empty:
        raise HTTPException(status_code=404, detail="Título no encontrado")
    
    #Obtengo los datos
    movie=movie.iloc[0]
    nombre=movie['title']
    año=movie['release_year']
    score=movie['popularity']

    return f'La película {nombre} fue estrenada en el año {año} con un score/popularidad de {score}'


#4. def votos_titulo( titulo_de_la_filmación ): Se ingresa el título de una filmación esperando como respuesta el título, la cantidad de votos y el valor promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, debemos contar con un mensaje avisando que no cumple esta condición y que por ende, no se devuelve ningun valor.
#Ejemplo de retorno: La película X fue estrenada en el año X. La misma cuenta con un total de X valoraciones, con un promedio de X

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):

    #Para evitar errores de ortografia lo estandarizamos
    titulo = titulo.lower()
    
    #Buscamos el titulo en el df y tambien lo pasamos a minusculas
    movie = df_api[df_api['title'].str.lower() == titulo.lower()]

    #Validacion
    if movie.empty:
        raise HTTPException(status_code=404, detail="Título no encontrado")
    
    #Obtengo los datos
    movie=movie.iloc[0]
    nombre=movie['title']
    año=movie['release_year']
    votos=movie['vote_count']
    promedio=movie['vote_average']

    return f'La película {nombre} fue estrenada en el año {año}. La misma cuenta con un total de {votos} valoraciones, con un promedio de {promedio}'

#5. def get_actor( nombre_actor ): Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha participado y el promedio de retorno. La definición no deberá considerar directores.
#Ejemplo de retorno: El actor X ha participado de X cantidad de filmaciones, el mismo ha conseguido un retorno de X con un promedio de X por filmación

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):

    #Para evitar errores de ortografia lo estandarizamos
    nombre_actor=nombre_actor.lower()

    #Busco las peliculas en las que aparece como actor y lo estandarizamos
    movies=df_api[df_api['cast_members'].apply(lambda x:nombre_actor in [y.lower() for y in x])]

    #Validacion
    if movies.empty:
        raise HTTPException(status_code=404, detail='Actor no encontrado')

    #Obtengo los datos 
    cantidad_filmaciones=movies.shape[0]
    retorno=round(movies['return'].sum(),2)
    retorno_promedio=round((retorno/cantidad_filmaciones),2)

    return f'El actor {nombre_actor} ha participado en {cantidad_filmaciones} películas, el mismo ha conseguido un retorno de {retorno} con un promedio de {retorno_promedio} por filmación'


#6.def get_director( nombre_director ): Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el éxito del mismo medido a través del retorno.
# Además, deberá devolver el nombre de cada película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.


@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):

    #Para evitar errores de ortografia lo estandarizamos
    nombre_director=nombre_director.lower()


    #Buscamos las peliculas que ha dirigido y lo estandarizamos
    movies=df_api[df_api['director'].str.lower()==nombre_director]

    #Validacion
    if movies.empty:
        raise HTTPException(status_code=404, detail='Director no encontrado')
    
    #Obtengo los datos 
    retorno=round(movies['return'].sum(),2)
    peliculas_dirigidas=movies[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')
    cantidad_filmaciones=movies.shape[0]

    return {'Director':nombre_director,
            'Cantidad de Películas':cantidad_filmaciones,
            'Retorno total': retorno,
            'Películas':peliculas_dirigidas}


#Modelo

df_modelo=pd.read_parquet('Data_Modelo')

# Crear una instancia de TfidfVectorizer con las stopwords en inglés
tfidf = TfidfVectorizer(stop_words='english')
# Crear la matriz tf-idf con los features de las películas
tfidf_matrix = tfidf.fit_transform(df_modelo['genres'] + ' ' + df_modelo['overview'] + ' ' + df_modelo['cast_members'] + ' ' + df_modelo['director'])

@app.get("/recomendacion/{title}")
def recomendacion(title: str):
    
    #Validacion de inicio
    if title not in df_modelo["title"].values:
           return {"Error": "Título no encontrado"}
    
    idx = df_modelo[df_modelo['title'] == title].index[0]
    #Calculo de la similitud del coseno
    cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    #Recopilacion de los scores de similitud en una lista de tuplas, donde el primer elemento es el índice y el segundo es el score.
    sim_scores = list(enumerate(cosine_sim))
    #Ordenar la lista de mayor a menor.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Obtener los 5 ítems más similares
    sim_scores = sim_scores[:6]

     #Excluir la primera película si es la misma que se ingresó
    if sim_scores[0][0] == idx:
        sim_scores = sim_scores[1:6]
    else:
        sim_scores = sim_scores[:5]
    # Obtener los títulos de los ítems recomendados y convertirlos en lista
    recommended_movies = df_modelo['title'].iloc[[i[0] for i in sim_scores]].tolist()

    result = {"Películas recomendadas": recommended_movies}

    # Devolver las películas recomendadas
    return result

    