import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import requests


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
#test
def show_song(uri_by_search):
    track_by_search = "https://open.spotify.com/embed/track/{}".format(uri_by_search)
    response = requests.get(track_by_search)
    res = response.json()["html"] 
    return res
#test

def page():
    title = "Song Recommendation Engine"
    st.title(title)

    st.write("First of all, welcome! This is the place where you can customize what you want to listen to based on genre and several key audio features. Try playing around with different settings and listen to the songs recommended by our system!")
    st.markdown("##")

    df_filter_name = pd.read_csv("song_recommendation-main/data/filter by name.csv")
    df_filter_lyrics = pd.read_csv("song_recommendation-main/data/filter by lyrics.csv")

    st.sidebar.markdown("Advance")
    select_event = st.sidebar.selectbox('How do you want to recommend for you',
                                    ['By Name', 'By Lyrics'])
    if select_event == "By Name":
        select_df = st.selectbox("Choose Music", df_filter_name)
        df_filter = df_filter_name.loc[(df_filter_name["name"] == select_df)]
        df_filter_uri = df_filter["uri"]
        df_filter_uri = df_filter_uri.values.tolist()
    if select_event == "By Lyrics":
        select_df = st.selectbox("Type Your Lyrics", df_filter_lyrics)
        df_filter = df_filter_lyrics.loc[(df_filter_lyrics["lyrics"] == select_df)]
        
        
    st.sidebar.markdown("")
    genre = st.sidebar.radio("", genre_names, index=genre_names.index("Pop"))
#     genre = st.sidebar.multiselect('',['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock'],['Electronic'])
    
    
      
        
    with st.container():
        col1, col2,col3,col4 = st.columns((15,0.5,0.5,0.5))
        with col1:
            #test
#             if df_filter_uri != null:
#                 res = show_song(df_filter_uri)
#                 components.html(res,height= 700)
             
            
            #test
            for i in df_filter_uri:
                test = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(i)
                components.html(test,height= 400)

            with st.expander("Advance"):
                st.markdown("***Choose features to make your own recommend list:***")
                start_year, end_year = st.slider(
                    'Select the year range',
                    1990, 2019, (int(df_filter['release_year']),int(df_filter['release_year']) - 1)
                )
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
