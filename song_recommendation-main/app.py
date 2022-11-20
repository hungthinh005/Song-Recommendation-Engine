import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import requests
import numpy as np
import string

st.set_page_config(page_title="Song Recommendation", layout="wide")
@st.cache(allow_output_mutation=True)
def load_data():
    df_filter_name = pd.read_csv("song_recommendation-main/data/filter by name.csv")
    df_filter_lyrics = pd.read_csv("song_recommendation-main/data/filter by lyrics.csv")

    df_filter_name['genres'] = df_filter_name.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    df_filter_lyrics['genres'] = df_filter_lyrics.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])

    exploded_track_df = df_filter_name.explode("genres")
    # exploded_track_df1 = df_filter_lyrics.explode("genres")
    return exploded_track_df

genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']
audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    genre = genre.lower()
#     [x.lower() for x in genre]
    genre_data = exploded_track_df[(exploded_track_df["genres"]==genre) 
                                   & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
   
    genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]


    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=len(genre_data), return_distance=False)[0]


    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()

    return uris, audios

def page():
    title = "Song Recommendation Engine"
    st.title(title)

    st.write("Welcome! This is the place where you can search and customize what you want to listen to based on genre and several key audio attributes. Enjoy!")
    st.markdown("##")

    df_filter_name = pd.read_csv("song_recommendation-main/data/filter by name.csv")
    df_filter_lyrics = pd.read_csv("song_recommendation-main/data/filter by lyrics.csv")

    st.sidebar.markdown("Advance")
    select_event = st.sidebar.selectbox('How do you want to recommend for you',
                                    ['By Name', 'By Lyrics'])
    if select_event == "By Name":
        select_df = st.selectbox("Type Your Music", df_filter_name)
        df_filter = df_filter_name.loc[(df_filter_name["name"] == select_df)]
        df_filter_uri = df_filter["uri"]
        df_filter_uri = df_filter_uri.values.tolist()
        df_filter_genre = df_filter["genres"]
        df_filter_genre = df_filter_genre.to_string(header=False, index=False)
#         df_filter_genre = df_filter_genre.translate(str.maketrans('', '', string.punctuation))
        for character in string.punctuation:
            if character != ",":
                df_filter_genre = df_filter_genre.replace(character, '')
        df_filter_playlist = df_filter["playlist"]
        df_filter_playlist = df_filter_playlist.to_string(header=False, index=False)
        df_filter_year = df_filter["release_date"]
        df_filter_year = df_filter_year.to_string(header=False, index=False)
        artists_name = df_filter["artists_name"]
        artists_name = artists_name.values[0]
        df_filter_artists = df_filter_name[(df_filter_name["artists_name"] == artists_name)]
        df_filter_artists.index = np.arange(1, len(df_filter_artists) + 1)
#         df_filter_by_artists = df_filter_by_artists.values.tolist()
    if select_event == "By Lyrics":
        select_df = st.selectbox("Type Your Lyrics", df_filter_lyrics)
        df_filter = df_filter_lyrics.loc[(df_filter_lyrics["lyrics"] == select_df)]
        df_filter_uri = df_filter["uri"]
        df_filter_uri = df_filter_uri.values.tolist()
        df_filter_genre = df_filter["genres"]
        df_filter_genre = df_filter_genre.to_string(header=False, index=False)
#         df_filter_genre = df_filter_genre.translate(str.maketrans('', '', string.punctuation))
        for character in string.punctuation:
            if character != ",":
                df_filter_genre = df_filter_genre.replace(character, '')
        df_filter_playlist = df_filter["playlist"]
        df_filter_playlist = df_filter_playlist.to_string(header=False, index=False)
        df_filter_year = df_filter["release_date"]
        df_filter_year = df_filter_year.to_string(header=False, index=False)
        artists_name = df_filter["artists_name"]
        artists_name = artists_name.values[0]
        df_filter_artists = df_filter_lyrics[(df_filter_lyrics["artists_name"] == artists_name)]
        df_filter_artists.index = np.arange(1, len(df_filter_artists) + 1)
        
    st.sidebar.markdown("")
    with st.expander("Choose your favorite genre"):
        genre = st.sidebar.radio("", genre_names, index=genre_names.index("Electronic"))
#     genre = st.sidebar.multiselect('',['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock'],['Electronic'])
                
    with st.container():
        col1, col2, col3 = st.columns((10, 10, 12))
        with col1:
            for i in df_filter_uri:
                show_song = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(i)
                components.html(show_song,height= 400)
            st.markdown("Advance")
        with col2:
            st.markdown("Genre: ")
            st.write("   ",df_filter_genre)
            st.markdown("Playlist: ")
            st.write("   ",df_filter_playlist)
            st.markdown("Release Date: ")
            st.write("   ",df_filter_year)
#             with st.expander("Choose features to make your own recommend list:"):
               
# #                 start_year, end_year = st.slider(
# #                     'Select the year range',
# #                     1990, 2019, (int(df_filter['release_year']),int(df_filter['release_year']) - 1)
# #                 )
#                 start_year, end_year = 1990, 2019
#                 acousticness = st.slider(
#                     'Acousticness',
#                     0.0, 1.0, float(df_filter['acousticness']))
#                 danceability = st.slider(
#                     'Danceability',
#                     0.0, 1.0, float(df_filter['danceability']))
#                 energy = st.slider(
#                     'Energy',
#                     0.0, 1.0, float(df_filter['energy']))
#                 instrumentalness = st.slider(
#                     'Instrumentalness',
#                     0.0, 1.0, float(df_filter['instrumentalness']))
#                 valence = st.slider(
#                     'Valence',
#                     0.0, 1.0, float(df_filter['valence']))
#                 tempo = st.slider(
#                     'Tempo',
#                     0.0, 244.0, float(df_filter['tempo']))
        with col3:
            df_filter_artists.style.set_properties(subset=["name"], **{'width': '700px'})
            df_filter_artists = df_filter_artists.rename(columns={"name": "                                           Setlist                                                                     "})
            st.write(df_filter_artists["                                           Setlist                                                                     "])
    
    with st.container():
        with st.expander("Choose features to make your own Recommendation List:"):

            start_year, end_year = 1990, 2019
            acousticness = st.slider(
                'Acousticness',
                0.0, 1.0, float(df_filter['acousticness']))
            danceability = st.slider(
                'Danceability',
                0.0, 1.0, float(df_filter['danceability']))
            energy = st.slider(
                'Energy',
                0.0, 1.0, float(df_filter['energy']))
            instrumentalness = st.slider(
                'Instrumentalness',
                0.0, 1.0, float(df_filter['instrumentalness']))
            valence = st.slider(
                'Valence',
                0.0, 1.0, float(df_filter['valence']))
            tempo = st.slider(
                'Tempo',
                0.0, 244.0, float(df_filter['tempo']))
    
    st.markdown("Recommended Songs")
    tracks_per_page = 10
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
    
    
    tracks = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)

    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
    
    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0
    
    with st.container():
        col1, col2, col3 = st.columns([2,1,2])
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page

        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio) in enumerate(zip(current_tracks, current_audios)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df_filter_name1 = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                            fig = px.line_polar(df_filter_name1, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)
            
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                        with st.expander("See more details"):
                            df_filter_name1 = pd.DataFrame(dict(
                            r=audio[:5],
                            theta=audio_feats[:5]))
                            fig = px.line_polar(df_filter_name1, r='r', theta='theta', line_close=True)
                            fig.update_layout(height=400, width=340)
                            st.plotly_chart(fig)

        else:
            st.write("No songs left to recommend")

page()
