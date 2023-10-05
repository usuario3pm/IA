
import spotipy 
import spotipy.util as util  #metodo de autentificacion
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import linear_kernel
import networkx as nx


class SpotipyClient():
    client = None
    client_id = None
    client_secret = None
    username = None
    redirect_uri = 'http://localhost:8080'
    
    def __init__(self, client_id, client_secret, username, redirect_uri, scope):
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.redirect_uri = redirect_uri
        self.scope = scope

    def client_auth(self):
        #              Autenticación API de Spotify
        token = util.prompt_for_user_token(self.username,self.scope,
            self.client_id,self.client_secret,self.redirect_uri)
        self.client = spotipy.Spotify(auth=token)

    def get_top_tracks(self):
        #       Obtener listado de pistas más escuchadas recientemente   
        top_tracks = self.client.current_user_top_tracks(time_range='short_term', limit=20)
        return top_tracks

    def create_tracks_dataframe(self, top_tracks):
        '''Obtener "audio features" de las pistas más escuchadas por el usuario'''
        tracks = top_tracks['items']
        tracks_ids = [track['id'] for track in tracks]
        audio_features = self.client.audio_features(tracks_ids)
        top_tracks_df = pd.DataFrame(audio_features)
        top_tracks_df = top_tracks_df[["id", "acousticness", "danceability", 
            "duration_ms", "energy", "instrumentalness",  "key", "liveness", 
            "loudness", "mode", "speechiness", "tempo", "valence"]]
        return top_tracks_df

    def get_artists_ids(self, top_tracks):
        '''Obtener ids de los artistas en "top_tracks"'''
        ids_artists = []

        for item in top_tracks['items']:
            artist_id = item['artists'][0]['id']
            artist_name = item['artists'][0]['name']
            ids_artists.append(artist_id)
        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists

    def get_similar_artists_ids(self, ids_artists):
        '''Expandir el listado de "ids_artists" con artistas similares'''
        ids_similar_artists = []
        for artist_id in ids_artists:
            artists = self.client.artist_related_artists(artist_id)['artists']
            for item in artists:
                artist_id = item['id']
                artist_name = item['name']
                ids_similar_artists.append(artist_id)

        ids_artists.extend(ids_similar_artists)

        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists

    def get_new_releases_artists_ids(self, ids_artists):
        '''Expandir el listado de "ids_artists" con artistas con nuevos lanzamientos'''

        new_releases = self.client.new_releases(limit=20)['albums']
        for item in new_releases['items']:
            artist_id = item['artists'][0]['id']
            ids_artists.append(artist_id)

        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))
        return ids_artists

    def get_albums_ids(self, ids_artists):
        '''Obtener listado de albums para cada artista en "ids_artists"'''
        ids_albums = []
        for id_artist in ids_artists:
            album = self.client.artist_albums(id_artist, limit=1)['items'][0]
            ids_albums.append(album['id'])
        return ids_albums

    def get_albums_tracks(self, ids_albums):
        '''Extraer 3 tracks para cada album en "ids_albums"'''
        ids_tracks = []
        for id_album in ids_albums:
            album_tracks = self.client.album_tracks(id_album, limit=1)['items']
            for track in album_tracks:
                ids_tracks.append(track['id'])
        return ids_tracks

    def get_tracks_features(self, ids_tracks):
        '''Extraer audio features de cada track en "ids_tracks" y almacenar resultado
        en un dataframe de Pandas'''

        ntracks = len(ids_tracks)

        if ntracks > 100:
            # Crear lotes de 100 tracks (limitacion de audio_features)
            m = ntracks//100
            n = ntracks%100
            lotes = [None]*(m+1)
            for i in range(m):
                lotes[i] = ids_tracks[i*100:i*100+100]

            if n != 0:
                lotes[i+1] = ids_tracks[(i+1)*100:]
        else:
            lotes = [ids_tracks]


        # Iterar sobre "lotes" y agregar audio features
        audio_features = []
        for lote in lotes:
            features = self.client.audio_features(lote)
            audio_features.append(features)

        audio_features = [item for sublist in audio_features for item in sublist]

        # Crear dataframe
        candidates_df = pd.DataFrame(audio_features)
        candidates_df = candidates_df[["id", "acousticness", "danceability", "duration_ms",
            "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", 
            "speechiness", "tempo", "valence"]]
        return candidates_df
    
    ####################################################################################################
    def compute_cossim(self, top_tracks_df, candidates_df):
        '''Calcula la similitud del coseno entre cada top_track y cada pista
        candidata en candidates_df. Retorna matriz de n_top_tracks x n_candidates_df'''
        top_tracks_mtx = top_tracks_df.iloc[:,1:].values
        candidates_mtx = candidates_df.iloc[:,1:].values

        # Estandarizar cada columna de features: mu = 0, sigma = 1
        scaler = StandardScaler()
        top_tracks_scaled = scaler.fit_transform(top_tracks_mtx)
        can_scaled = scaler.fit_transform(candidates_mtx)

        # Normalizar cada vector de características (magnitud resultante = 1)
        top_tracks_norm = np.sqrt((top_tracks_scaled*top_tracks_scaled).sum(axis=1))
        can_norm = np.sqrt((can_scaled*can_scaled).sum(axis=1))

        n_top_tracks = top_tracks_scaled.shape[0]
        n_candidates = can_scaled.shape[0]
        top_tracks = top_tracks_scaled/top_tracks_norm.reshape(n_top_tracks,1)
        candidates = can_scaled/can_norm.reshape(n_candidates,1)

        # Calcular similitudes del coseno
        cos_sim = linear_kernel(top_tracks,candidates)
        return cos_sim
    ################################################################################################################3

###############################################################################################################
    def create_similarity_graph(cosine_similarity_matrix, top_track_names, candidate_track_names, humbral=0.5):
        """
        Crea un grafo de similitud basado en una matriz de similitud del coseno.

        Args:
            cosine_similarity_matrix (numpy.ndarray): La matriz de similitud del coseno.
            top_track_names (list): Lista de nombres de pistas principales.
            candidate_track_names (list): Lista de nombres de pistas candidatas.
            threshold (float): Umbral para la similitud del coseno, solo se agregan aristas si supera este umbral.

        Returns:
            networkx.Graph: El grafo de similitud con nodos y aristas.
        """
        Candidate_gf = nx.Graph()

        # Agregar nodos al grafo
        for top_track in top_track_names:
            Candidate_gf.add_node(top_track, type='top_track')

        for candidate_track in candidate_track_names:
            Candidate_gf.add_node(candidate_track, type='candidate_track')

        num_top_tracks = len(top_track_names)
        num_candidates = len(candidate_track_names)

        # Agregar aristas basadas en similitud del coseno
        for i in range(num_top_tracks):
            for j in range(num_candidates):
                similarity = cosine_similarity_matrix[i, j]
                if similarity > humbral:
                   Candidate_gf.add_edge(str(top_track_names[i]), str(candidate_track_names[j]), similarity=similarity)
        print("ok")
        return Candidate_gf
    
    def bfs_search_in_graph(self, graph, start_track_name, content_threshold=0.9):
        """
        Realiza una búsqueda basada en contenido en el grafo de similitud utilizando Breadth-First Search (BFS).

        Args:
            graph (networkx.Graph): El grafo de similitud.
            start_track_name (str): El nombre de la pista principal desde la cual comenzar la búsqueda.
            content_threshold (float): Umbral para la similitud basada en contenido.

        Returns:
            list: Lista de pistas similares basadas en contenido encontradas durante la búsqueda.
        """
        similar_tracks = []

        # Crear una cola para el BFS
        queue = [(start_track_name, None)]  # Tupla (nombre de la pista, padre)

        while queue:
            current_track, parent_track = queue.pop(0)

            # Verificar si la pista actual es una pista candidata
            if 'type' in graph.nodes[current_track] and graph.nodes[current_track]['type'] == 'candidate_track':
                # Realizar el filtrado basado en contenido aquí
                top_track_features = self.create_tracks_dataframe(self.get_top_tracks())
                candidate_track_features = self.get_tracks_features([current_track])
                content_similarity = self.compute_cossim(top_track_features, candidate_track_features)
                if content_similarity >= content_threshold:
                    similar_tracks.append(current_track)

            # Agregar vecinos no visitados a la cola
            neighbors = graph.neighbors(current_track)
            for neighbor in neighbors:
                if neighbor != parent_track:
                    queue.append((neighbor, current_track))

        return similar_tracks

"""
    def content_based_filtering(self, pos, cos_sim, ncands, umbral = 0.8):
        '''Dada una pista de top_tracks (pos = 0, 1, ...) extraer "ncands" candidatos,
        usando "cos_sim" y siempre y cuando superen un umbral de similitud'''

        # Obtener todas las pistas candidatas por encima del umbral
        idx = np.where(cos_sim[pos,:]>=umbral)[0] # ejm. idx: [27, 82, 135]

        # Y organizarlas de forma descendente (por similitudes de mayor a menor)
        idx = idx[np.argsort(cos_sim[pos,idx])[::-1]]

        # Si hay más de "ncands", retornar únicamente un total de "ncands"
        if len(idx) >= ncands:
            cands = idx[0:ncands]
        else:
            cands = idx
        return cands
    
        
"""
