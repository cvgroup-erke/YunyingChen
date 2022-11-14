# Subtitle Detection By Yolov3


## Environment
```commandline
pip install -r requirement.txt
```

## Dataset
Download the dataset with the following link and save the .zip file under ./data/ folder.               
链接：https://pan.baidu.com/s/1M7BaXziPVsXvyZSzbUt42A?pwd=ksc3                   
提取码：ksc3                      
```
cd data/
sh DataGen.sh
```
It will generate around 41,000 images with subtitles and split the dataset into training set and validation set by 80% and 20%.


        

## Train             
```commandline
cd ../
python train.py
```
 The models will be saved in checkpoints folder.                

## Inference           
Put the images into testimg folder and the results will be saved in output folder.
```commandline
python detect.py  --weights_path checkpoints/your_model_path.pth
```
