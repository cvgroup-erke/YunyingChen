## BBC_class

### Intro
This is a color correction model by using Gan and Conditional Gan.       
In Discriminator, it is a convolutional network using wgan-loss and gradient penalty.     
In Generator, it is a U-Net like network resizing the img into 256, 128 and 64. Its loss includes adversarial loss, perceptual loss and pixel loss.

### Performance
Original Imgs:                     
![ori](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_360.jpg)
![ori](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_14580.jpg)
![ori](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_22680.jpg)




Gan:              
![gan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_360_gan.jpg)
![gan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_14580_gan.jpg)
![gan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_22680_gan.jpg)



Conditional Gan:                           
![cgan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_360_cgan.jpg)
![cgan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_14580_cgan.jpg)
![cgan](https://github.com/cvgroup-erke/YunyingChen/blob/main/BBC_class/imgs/BBC_Western_22680_cgan.jpg)
