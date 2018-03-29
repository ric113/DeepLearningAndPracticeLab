#!/bin/sh

python3 main.py 
python3 main.py --layer 56
python3 main.py --layer 110

python3 main.py --model CNN
python3 main.py --model CNN --layer 56
python3 main.py --model CNN --layer 110