# YunyingChen's HomeWork




## Homework1
-证件模糊检测方案.pdf                           

## Homework2              
Eliminated the 512*512 anchor boxes by changing the code in config.py and faceboxes.py                                                 
                             
## Homework3           
nms.py contains the nms and soft-nms function implemented in python                            

## GAN               
In this repo, the network has been modified by using ResNet50 as backbone.                        
It shows the 3 results by using different losses:
1. using original loss      
2. using Wgan loss with clipping to limit the parameters to [-0.01, 0.01]             
3. using Wgan with gradient penalty                   
