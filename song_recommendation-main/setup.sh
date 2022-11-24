mkdir -p ~/song_recommendation-main/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~song_recommendation-main/config.toml
