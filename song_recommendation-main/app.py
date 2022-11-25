import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import string
from PIL import Image


st.set_page_config(page_title="Song Recommendation", layout="wide")
@st.cache(allow_output_mutation=True)
def load_data():
    df_filter_name = pd.read_csv("song_recommendation-main/data/filter by name.csv")
    df_filter_lyrics = pd.read_csv("song_recommendation-main/data/filter by lyrics.csv")

    df_filter_name['genres'] = df_filter_name.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])
    df_filter_lyrics['genres'] = df_filter_lyrics.genres.apply(lambda x: [i[1:-1] for i in str(x)[1:-1].split(", ")])

    exploded_track_df = df_filter_name.explode("genres")
    # exploded_track_df1 = df_filter_lyrics.explode("genres")
    # exploded_track_df = df_filter_name
    return exploded_track_df

# genre_names = ['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock']

audio_feats = ["acousticness", "danceability", "energy", "instrumentalness", "valence", "tempo"]

exploded_track_df = load_data()
genre_names = pd.unique(exploded_track_df['genres'])

def n_neighbors_uri_audio(genre, start_year, end_year, test_feat):
    # genre = genre.lower()
    # [x.lower() for x in genre]

    genre = pd.Series(genre)
    genre_data = exploded_track_df[(exploded_track_df["genres"].isin(genre)) 
                                   & (exploded_track_df["release_year"]>=start_year) & (exploded_track_df["release_year"]<=end_year)]
   
    # genre_data = genre_data.sort_values(by='popularity', ascending=False)[:500]
    # genre_data = genre_data[:,~np.all(np.isnan(genre_data), axis=0)]
    genre_data = genre_data.drop_duplicates(subset=['uri'])[:500]
    neigh = NearestNeighbors()
    neigh.fit(genre_data[audio_feats].to_numpy())
    length = len(genre_data)

    n_neighbors = neigh.kneighbors([test_feat], n_neighbors=length, return_distance=False)[0]


    uris = genre_data.iloc[n_neighbors]["uri"].tolist()
    audios = genre_data.iloc[n_neighbors][audio_feats].to_numpy()

    return uris, audios
def page():
    
    st.markdown("<h1 style='text-align: center; color: white;'>Song Recommendation Engine</h1>", unsafe_allow_html=True)
    st.write("Welcome! This is the place where you can search and customize what you want to listen to based on genre and several key audio attributes. Enjoy!")
    st.markdown("##")

    df_filter_name = pd.read_csv("song_recommendation-main/data/filter by name.csv")
    df_filter_lyrics = pd.read_csv("song_recommendation-main/data/filter by lyrics.csv")
    
    image = Image.open('song_recommendation-main/data/music.png')
    st.sidebar.image(image)
#     st.sidebar.markdown("**Advance**")
    select_event = st.sidebar.selectbox('How do you want to recommend for you:',
                                    ['By Name', 'By Lyrics'])
    if select_event == "By Name":
        select_df = st.selectbox("Type Your Music", df_filter_name)
        df_filter = df_filter_name.loc[(df_filter_name["name"] == select_df)]
        df_filter_uri = df_filter["uri"]
        df_filter_uri = df_filter_uri.values.tolist()
        df_filter_genre = df_filter["genres"]
        df_filter_genre = df_filter_genre.to_string(header=False, index=False)
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
    if select_event == "By Lyrics":
        select_df = st.selectbox("Type Your Lyrics", df_filter_lyrics)
        df_filter = df_filter_lyrics.loc[(df_filter_lyrics["lyrics"] == select_df)]
        df_filter_uri = df_filter["uri"]
        df_filter_uri = df_filter_uri.values.tolist()
        df_filter_genre = df_filter["genres"]
        df_filter_genre = df_filter_genre.to_string(header=False, index=False)
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
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    
    with st.sidebar.markdown(""):
#         with st.expander("Choose your favorite genre"):
        default1 = ['big room', 'edm']
        genre = st.multiselect("Choose Your Favorite Genre:", genre_names, default = default1)

        # genre = st.selectbox("Choose your favorite genre:",['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock'])
#     genre = st.sidebar.multiselect('',['Dance Pop', 'Electronic', 'Electropop', 'Hip Hop', 'Jazz', 'K-pop', 'Latin', 'Pop', 'Pop Rap', 'R&B', 'Rock'],['Electronic'])
#         if "counter" not in st.session_state:
#             st.session_state.counter = 1 
        
        
                            
#             st.session_state.counter += 1
#             components.html(
#                 f"""
#                     <p>{st.session_state.counter}</p>
#                     <script>
#                         window.parent.document.querySelector('section.main').scrollTo(100, 1000);
#                     </script>
#                 """,
#                 height=0
#             )
    st.sidebar.markdown("<a href='#link_to_list'><h2>Get List</h2></a>", unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns((10, 10, 12))
        with col1:
            for i in df_filter_uri:
                show_song = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(i)
                components.html(show_song,height= 400)
        
        with col2:
            st.markdown("**Genre:** ")
            st.markdown(df_filter_genre)
            st.markdown("**Playlist:** ")
            st.markdown(df_filter_playlist)
            st.markdown("**Release Date:** ")
            st.markdown(df_filter_year)
        with col3:
            df_filter_artists.style.set_properties(subset=["name"], **{'width': '700px'})
            df_filter_artists = df_filter_artists.rename(columns={"name": "                                                   Setlist                                                                        "})
            st.write(df_filter_artists["                                                   Setlist                                                                        "])
    st.markdown("<h2 style='text-align: center; color: white;'>Advance</h2>", unsafe_allow_html=True) 
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

#     st.markdown("Recommended Songs")
    tracks_per_page = 10
    test_feat = [acousticness, danceability, energy, instrumentalness, valence, tempo]
    uris, audios = n_neighbors_uri_audio(genre, start_year, end_year, test_feat)
    
    
    tracks = []
    list_uri = []
    for uri in uris:
        track = """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""".format(uri)
        tracks.append(track)
        uri1 = uri
        list_uri.append(uri1)
    if 'previous_inputs' not in st.session_state:
        st.session_state['previous_inputs'] = [genre, start_year, end_year] + test_feat
    
    current_inputs = [genre, start_year, end_year] + test_feat
    if current_inputs != st.session_state['previous_inputs']:
        if 'start_track_i' in st.session_state:
            st.session_state['start_track_i'] = 0
        st.session_state['previous_inputs'] = current_inputs

    if 'start_track_i' not in st.session_state:
        st.session_state['start_track_i'] = 0

  
    st.markdown("<div id='link_to_list'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Your Recommendation List</h2>", unsafe_allow_html=True)   
    with st.container():
        col1, col2, col3, col4 = st.columns([1.7,1.4,1.7,1.4])
        
        current_tracks = tracks[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        
        current_audios = audios[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        
        current_uri = list_uri[st.session_state['start_track_i']: st.session_state['start_track_i'] + tracks_per_page]
        if st.session_state['start_track_i'] < len(tracks):
            for i, (track, audio, uri1) in enumerate(zip(current_tracks, current_audios, current_uri)):
                if i%2==0:
                    with col1:
                        components.html(
                            track,
                            height=400,
                            
                        )
                        
                    with col2:
                        with st.expander("Details Features"):
                                df_filter_name1 = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                                fig = px.line_polar(df_filter_name1, r='r', theta='theta', line_close=True)
                                fig.update_layout(height=100, width=280,margin=dict(l=40, r=40, b=100, t=10))
                                st.plotly_chart(fig, theme='streamlit')     
                        with st.expander("Song's Info"):              
                            df_filter_name_for_list = df_filter_name[df_filter_name['uri'] == uri1]
                            df_filter_name_for_list['No'] = df_filter_name_for_list['genres'].apply(lambda n: len(n.split(',')))
                            df_filter_genre1 = df_filter_name_for_list[['genres','No']]
                            
                            
                            df_filter_genre1['genres'] = df_filter_name_for_list['genres'].str.replace("'",'')
                            df_filter_genre1['genres'] = df_filter_genre1['genres'].str.replace("[",'')
                            df_filter_genre1['genres'] = df_filter_genre1['genres'].str.replace("]",'')
                            
                            
                            df_filter_genre1.set_index('No', inplace = True)
                            
                            st.dataframe(df_filter_genre1)
                            
                            st.markdown("**Playlist:** ")
                            df_filter_playlist1 = df_filter_name_for_list['playlist']
                            df_filter_playlist1 = df_filter_playlist1.to_string(header=False, index=False)
                            st.markdown(df_filter_playlist1)
                            st.markdown("**Release Date:** ")
                            df_filter_year1 = df_filter_name_for_list['release_date']
                            df_filter_year1 = df_filter_year1.to_string(header=False, index=False)
                            st.markdown(df_filter_year1)
                        temp = ''
                        height_value = 200
                        components.html(
                            temp,
                            height = height_value,
                        )
                               
                else:
                    with col3:
                        components.html(
                            track,
                            height=400,
                        )
                    with col4:
                        with st.expander("Details Features"):
                                df_filter_name1 = pd.DataFrame(dict(
                                r=audio[:5],
                                theta=audio_feats[:5]))
                                fig = px.line_polar(df_filter_name1, r='r', theta='theta', line_close=True)
                                fig.update_layout(height=280, width=280,margin=dict(l=40, r=40, b=40, t=40))
                                st.plotly_chart(fig, theme='streamlit')  
                        with st.expander("Song's Info"):               
                            df_filter_name_for_list = df_filter_name[df_filter_name['uri'] == uri1]
                            df_filter_name_for_list['No'] = df_filter_name_for_list['genres'].apply(lambda n: len(n.split(',')))
                            df_filter_genre1 = df_filter_name_for_list[['genres','No']]
                            
                            
                            df_filter_genre1['genres'] = df_filter_name_for_list['genres'].str.replace("'",'')
                            df_filter_genre1['genres'] = df_filter_genre1['genres'].str.replace("[",'')
                            df_filter_genre1['genres'] = df_filter_genre1['genres'].str.replace("]",'')
                            
                            
                            df_filter_genre1.set_index('No', inplace = True)
                            
                            st.dataframe(df_filter_genre1)
                            
                            st.markdown("**Playlist:** ")
                            df_filter_playlist1 = df_filter_name_for_list['playlist']
                            df_filter_playlist1 = df_filter_playlist1.to_string(header=False, index=False)
                            st.markdown(df_filter_playlist1)
                            st.markdown("**Release Date:** ")
                            df_filter_year1 = df_filter_name_for_list['release_date']
                            df_filter_year1 = df_filter_year1.to_string(header=False, index=False)
                            st.markdown(df_filter_year1)
                        temp = ''
                        height_value = 200
                        components.html(
                            temp,
                            height = height_value,
                        )      

        else:
            st.write("No songs left to recommend")
        if st.button("Recommend More Songs"):
            if st.session_state['start_track_i'] < len(tracks):
                st.session_state['start_track_i'] += tracks_per_page
page()
