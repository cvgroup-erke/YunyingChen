# YunyingChen's HomeWork




## Homework1
-证件模糊检测方案.pdf        
-Faceboxes                
![AFW](https://github.com/cvgroup-erke/YunyingChen/blob/main/HMK1/Faceboxes/AFW_Result.png "AFW=98.44%")
![PASCAL](https://github.com/cvgroup-erke/YunyingChen/blob/main/HMK1/Faceboxes/PASCAL_Result.png "PASCAL=96.28%")


## Homework2             
Eliminated the 512*512 anchor boxes by changing the code in config.py and faceboxes.py                            
Here is the result:                 
![AFW](https://github.com/cvgroup-erke/YunyingChen/blob/main/HMK2/Faceboxes/AFW.png "AFW=98.12%")
![PASCAL](https://github.com/cvgroup-erke/YunyingChen/blob/main/HMK2/Faceboxes/PASCAL.png "PASCAL=96.55%")
                             
## Homework3           
nms.py contains the nms and soft-nms function implemented in python          

## GAN       
In this repo, the network has been modified by using ResNet50 as backbone.            
The loss of Discriminator is 0.5 and the loss of Generator is 0.                 
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss.png)         
The generated images are as follows:                  
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/9600.png)
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/10000.png)