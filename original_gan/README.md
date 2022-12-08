## GAN

The Discriminator and Generator network are defined in resnet50.py. ResNet50 is used as backbone in Descriminator and Generator.            
The 1-dimension random noise input is changed into a 3-dimension input [3,7,7].                



Here are some problems that I have encountered and I have some solutions to them, while training:
1. If the Discriminator is too big, the loss of Discriminator will easily reach 0 and the loss of generator will always stay high. So I tried to keep the Discriminator network small and the Generator network big.
2. If the random noise input is too big, the Generator network is hard to train and will gain a high loss. So I tried to keep the 3-dimension input small.  

For the final losses, the loss of Discriminator reached 50 and the loss of Generator reached 0.       
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss.png)      
The generated images are as follows:      
![ouput](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/9600.png)
![output](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/10000.png)

But here are some problems that are unsolved:                
During the training process, the losses of Generator and Discriminator will jump from a sudden and the result will suddenly turn bad while the learning rate is really small.         
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss1.png)  
![loss](https://github.com/cvgroup-erke/YunyingChen/blob/main/original_gan/imgs/loss2.png)   
