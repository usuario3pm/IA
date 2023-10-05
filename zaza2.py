
from zaza import*
from tabulate import tabulate
from art import *


# función para esperar
def Esperar():
    input("\nPresiona Enter para continuar...")

p = text2art("Proyecto  ", font="big")
ia = text2art("IA",font="block")
print(p,ia)

Esperar()
###################################################
# Se crea una instancia de la clase SpotipyClient del modulo backend
clien_id = '9542debab94c44aaba5d3ec2ceea10cd' #aplicacion1
#clien_id ='bb5284de43f54f2aadff6b8f174d88d4' #aplicacion2
client_secret = '66a5b895bc934b77a8d17a4adce9a0d4'#aplicacion1
#client_secret = '3dabffd21c5a450ea71461711c76cc2d'#aplicacion2

user_name ='31j2eafa55r2ofbce6aiyqjrlmei'
redirect_uri = 'http://localhost:8080'
scope = 'playlist-modify-private,playlist-modify-public,user-top-read'
init = SpotipyClient(clien_id,client_secret,user_name,redirect_uri,scope)
#autenticamos al usuario 
init.client_auth()
top_20 = init.get_top_tracks()
#mostrar  las ultimas 20 caciones escuchadas por el/la usuari@
for i, item in enumerate(top_20['items']):
            print(f"""      {i+1}: {item['name']} <--> {item['artists'][0]['name']} """)
Esperar()
dataframe_top_20 = init.create_tracks_dataframe(top_20)
#imprimir el dataframe con las ultimas 20 caciones
print(tabulate(dataframe_top_20, headers='keys', tablefmt='pretty'))
#Esperar()
ids_artistas = init.get_artists_ids(top_20)
ids_artistas = init.get_similar_artists_ids(ids_artistas)
ids_artistas = init.get_new_releases_artists_ids(ids_artistas)
#print('1\n')
print(ids_artistas)
#Esperar()
ids_albums = init.get_albums_ids(ids_artistas)
#print('2\n')
print(ids_albums)
#Esperar()
ids_pista = init.get_albums_tracks(ids_albums)
#print('3\n')
print(ids_pista)
#Esperar()
#                        pistas candidatas 
pistas_candidatas_df = init.get_tracks_features(ids_pista)
print(tabulate(pistas_candidatas_df, headers='keys', tablefmt='pretty'))
df_cos_simil = init.compute_cossim(dataframe_top_20,pistas_candidatas_df)
print(df_cos_simil)
# Crear un grafo dirigido desde el DataFrame de cadidatos
grafo_candidatas = init.create_similarity_graph(df_cos_simil, top_20, ids_pista)
