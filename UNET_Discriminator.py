import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UNET,self).__init__()

        #ENCODER ----
        #The choice of 64 is a reasonable starting point for the number of filters in the first layer, as it captures basic features like edges, textures, and patterns.
        self.enc1 = self.conv_block(in_channels,64)

        #After the first layer, the model will downsample the feature map (reduce its spatial dimensions using pooling), and at the same time, it learns more abstract features.
        self.enc2 = self.conv_block(64, 128)
        
        #As you go deeper into the encoder, the network learns increasingly abstract and complex representations of the data. 
        self.enc3 = self.conv_block(128, 256)

        #At this point, the network has learned quite complex features, and its spatial resolution is reduced. The number of channels is increased to 512 to capture very high-level representations.
        self.enc4 = self.conv_block(256, 512)



 
        #Bottleneck -----
        #The bottleneck is the central part of the network where the feature map is at its smallest spatial size (after several downsampling steps). At this stage, the model has a lot of high-level features but in a very compressed form.
        #The number of channels is increased to 1024 to allow the model to capture very complex and high-dimensional features before it starts to upsample and reconstruct the image in the decoder.
        self.bottleneck =self.conv_block(512, 1024)



        #Decoder ---
        #In the decoder, the model begins to reconstruct the original image, starting from the most abstract features. It upsamples the feature map to the next higher resolution (increasing spatial dimensions). The number of channels is reduced to 512, which corresponds to a less complex representation, as the model begins to combine high-level features from the bottleneck with spatial information.
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        #As the feature map continues to upsample, the model reduces the number of channels to 256, again focusing on combining lower-level features with spatial information.
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(1024, 256)

        #The number of channels is further reduced to 128, allowing the model to focus on finer details and combine them with spatial information.
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 128)

        #Finally, the feature map is upsampled again, and the number of channels is reduced to 64. At this stage, the model focuses on refining the image.
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(256, 64)

        #Final Layer
        self.final_conv = nn.Conv2d(64, out_channels,kernel_size=1)



    #Increased Representation Power: Two consecutive convolutional layers allow the network to learn more complex features.
    #Non-linearity: By adding the ReLU activation after each convolution, the model can learn more complex, non-linear relationships in the data.
    #Efficient Learning: Using multiple convolutions helps the network learn hierarchical features. The first layer might learn basic features like edges and textures, and subsequent layers combine these features to detect more complex patterns (shapes, objects, etc.).
    def conv_block(self,in_channels, out_channels):
        return(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )


    #The forward() method defines the actual forward pass of the U-Net model, where data flows through the various layers of the network, from the input to the final output.
    def forward(self, x):
        

        #Encoder (FORWARD)
        #In the encoder section, the input passes through a series of convolutional blocks (which are defined earlier in the __init__() method) with pooling applied to reduce the spatial dimensions of the feature map.
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        #Bottleneck (FORWARD)        
        #The bottleneck is the deepest layer of the U-Net, where the feature maps are highly abstracted, and spatial resolution is minimized. Here, the feature map is processed without any further spatial reduction.
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        #Decoder (FORWARD) 
        #The decoder section is responsible for reconstructing the original image by upsampling the feature map and reducing the number of channels gradually, bringing the spatial dimensions back to the original size.
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([dec4,enc4], dim=1) 
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        

        #After passing through the decoder, the output feature map has the required spatial resolution, and the number of channels is reduced to the desired output channels (e.g., 3 for RGB images).
        return self.final_conv(dec1)





class Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Discriminator, self).__init__()
        

        #The nn.Sequential container in PyTorch allows you to create a model by stacking layers in a sequence. It will automatically apply each layer to the input in the order they are defined. This is a compact and efficient way to define a network where the flow of data is straightforward from one layer to the next.
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            #nn.Sigmoid()
        )

    #The Discriminator is part of a Generative Adversarial Network (GAN), and it is used to distinguish between real and fake images. The forward method defines how the input image passes through the network to output a probability indicating whether the image is real or fake.
    def forward(self, x):
        return self.model(x)