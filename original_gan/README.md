## GAN

The Discriminator and Generator are defined in resnet50.py. They are using ResNet50 as backbone.            
The 1-dimension random noise input is changed into a 3-dimension input [3,7,7]                




Here are some problems that I have encountered and made some solutions to them, while training:
1. If the Discriminator is too big, the loss of Discriminator will easily reach 0 and the loss of generator will always stay high. So I tried to keep the Discriminator network small and the Generator network big.
2. If the random noise input is too big, the Generator network is hard to train and will gain a high loss. So I try to keep the 3-dimension input small.  

For the final loss, the loss of Discriminator can reach 0.5 and the loss of Generator can reach 0.       
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss.png)      
The generated images are as follows:      
![ouput](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/9600.png)
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/10000.png)

But here are some problems are unsolved:                
While training the loss of Generator and Discriminator will jump from a sudden and the result will suddenly will bad.         
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss1.png)  
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss2.png)   
