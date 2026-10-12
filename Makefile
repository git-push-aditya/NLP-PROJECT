PY=.venv/bin/python
smoke:
	cd src && ../$(PY) s00_sample.py --smoke && ../$(PY) s02_translate.py --smoke && ../$(PY) s01_lexical.py --smoke && ../$(PY) s03_embed.py --smoke && ../$(PY) s04_evaluate.py --smoke
