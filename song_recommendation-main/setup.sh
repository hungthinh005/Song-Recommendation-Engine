mkdir -p ~song_recommendation-main/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~song_recommendation-main/.streamlit/config.toml
