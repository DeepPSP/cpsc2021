#!/bin/sh
black . --extend-exclude .ipynb -v --exclude "/(build|dist|torch\_ecg|official_entry|entry\_2021\.py|score\_2021\.py|pantompkins\.py)/"
flake8 . --count --ignore="E501 W503 E203 F841 E402 E731" --show-source --statistics --exclude=./.*,build,dist,torch_ecg,official_entry,entry_2021.py,score_2021.py,pantompkins.py,*.ipynb
