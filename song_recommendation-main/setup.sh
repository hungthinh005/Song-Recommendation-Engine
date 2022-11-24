mkdir -p ~/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
" > ~song_recommendation-main/config.toml
