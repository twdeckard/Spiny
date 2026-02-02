# Moltbot Spiny

Corpus:
- /mnt/moltbot/corpus/images

Local state (NOT in git):
- /ssd/moltbot/thumbs
- /ssd/moltbot/qdrant
- /ssd/moltbot/meta.sqlite
- /ssd/moltbot/venv

Run:
- source /ssd/moltbot/venv/bin/activate
- python3 index_images.py
- python3 search_images.py "query"
