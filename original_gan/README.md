## GAN

The Discriminator and Generator are defined in Resnet.py. They are using Resnet50 as backbone.            
The 1-dimension random noise input is changed into a 3-dimension input [3,7,7]                




Here are some problems that I have encountered and I made some solutions to them, while training:
1. If the Discriminator is too big, the loss of Discriminator will easily reach 0 and the loss of generator will always stay high. So I tried to keep the Discriminator network small and the Generator network big.
2. If the random noise input is too big, the Generator network is hard to train and will gain a high loss. So I try to keep the 3-dimension input small.  

But here are some problems are unsolved:                
While training the loss of Generator and Discriminator will jump from a sudden and the result will suddenly turn bad.         
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss1.png)  
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss2.png)   
