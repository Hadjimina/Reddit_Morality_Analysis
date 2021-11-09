#!/bin/bash
pip3 -r requirements.txt || pip -r requirements.txt
python3 -m spacy download en_core_web_trf || python -m spacy download en_core_web_trf
cd datasets
python3 get_data.py
