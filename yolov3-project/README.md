# Subtitle Detect by YOLOV3




## DATA AUGMENTATION
Around 2,000 images are with subtitles cutting from the videos. Cutting out all the subtitles from the images and adding them into the non-subtitle images can expand the dataset into almost 39,000. The dataset is seperated into training set and validation set by 80% and 20%.
![subtitle](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/data/subtitle/subtitles.jpg)                    
![non](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/data/subtitle/images.jpg)                      
![mix](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/data/subtitle/NewSample.jpg)                                




## NETWORK    
This network detects 2 classes from the images and is only with one layer as its output. The anchor is modified from 3 to 6.                     
More details please check config/yolov3.cfg 


                             
## PERFORMANCE    
The mAp for validation set is 0.99974 in 3 epoches.                                  
![image1](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/output/BBC_China_17600_1104.png)   
![image2](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/output/BBC_China_18980_270.png)   
![image3](https://github.com/cvgroup-erke/YunyingChen/blob/main/yolov3-project/output/BBC_China_20480_1195.png)   
