#!/bin/sh

# scp -P 30006 ./models/OldModel.py dl2018@140.113.215.195:~/Lab3/ImageCaptioning.pytorch/models/
scp -P 30006 ./models/AttModel.py dl2018@140.113.215.195:~/Lab3/ImageCaptioning.pytorch/models/
scp -P 30006 ./eval_utils.py dl2018@140.113.215.195:~/Lab3/ImageCaptioning.pytorch/
scp -P 30006 ./eval.py dl2018@140.113.215.195:~/Lab3/ImageCaptioning.pytorch/