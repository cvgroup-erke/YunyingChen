#!/bin/bash

unzip data.zip 

#generate new data by mixing up subtitle and non-subtitle images
echo 'Data Generating!'
python MixUp.py

#Get all the dataset and split them into training set and validation set
python GetDataset.py

echo 'Data Generated!'