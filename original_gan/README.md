## GAN

The Discriminator and Generator network are defined in resnet50.py. ResNet50 is used as backbone in Descriminator and Generator.            
The 1-dimension random noise input is changed into a 3-dimension input [3,7,7].                


 
Using the original loss in Gan, the generated images are as follows:      
![ouput](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/ori_gan1.png)
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/ori_gan2.png)



Changing the loss to Wgan, the generated images are as follows:             
![ouput](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/wgan1.png)
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/wgan2.png)
