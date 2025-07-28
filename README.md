to recreate the original image. Our method, like all autoencoders, 
consists of an encoder that converts the observed image to a 
latent representation and a decoder that uses the latent 
representation to reconstruct the original signal. We use an 
asymmetric design, in contrast to classical autoencoders, which 
enables the encoder to work on input images and a lightweight 
decoder to reconstruct the entire image from the latent 
representation. An asymmetric encoder-decoder architecture 
can be particularly advantageous. For example, in image 
segmentation tasks, a complex encoder can be employed to 
deeply analyze and understand the nuances of input images, 
capturing detailed features and textures. This might involve 
layers that progressively downsample the image while 
increasing the depth of features extracted. On the other hand, the 
decoder, which is responsible for generating the segmented 
image, might be simpler or structured differently. It needs to 
upsample the feature map to the original image size and classify 
each pixel, but this process might not require as intricate an 
understanding as the initial feature extraction. This asymmetry 
allows for a more efficient processing where the encoder focuses 
on detailed feature extraction and the decoder on accurately 
mapping these features back to the spatial dimensions of the 
original image. Thereafter, we utilize the learned weights in the 
encoder to employ transfer learning with our limited training 
samples, which were acquired from observing high-resolution 
satellite images. Subsequently, we perform classification to 
classify each pixel. 
A. Pre-text Stage 
In our proposed architecture, the encoder is founded on 
ResNet architecture, which is adept at handling complex image 
data. This encoder processes the input through a sequence of 
residual blocks, a methodology renowned for its efficacy in 
preserving information across layers in deep neural networks. 
Intriguingly, our encoder network is designed to accommodate 
dual inputs: one catering to the spatial characteristics of the 
image, and the other, a more straightforward fully-connected 
encoder, is dedicated to processing the spectral data associated 
with the patch's central region. 
Fig. 1. Block diagram of proposed method. 
The spatial input undergoes a patchification process, where 
the original image is cropped into patches of 32by32 pixels. This 
granular approach allows for a more detailed and localized 
analysis of the image, facilitating the extraction of finer features. 
Concurrently, the spectral data, processed through the fully
connected encoder, provides complementary information that 
enriches the understanding of the patch's overall context, 
particularly in terms of its spectral characteristics. 
At a subsequent stage, known as the latent stage, the outputs 
of the ResNet and the fully-connected encoder are fused. This 
fusion creates a comprehensive latent representation that 
encapsulates both the detailed spatial features and the broader 
Authorized licensed use limited to: University of Szeged. Downloaded on April 27,2024 at 12:50:05 UTC from IEEE Xplore.  Restrictions apply. 
spectral information. In the decoding phase, we utilize a decoder 
inspired by ResNet-18, a more compact variant of the ResNet 
architecture. This modified decoder is designed to reconstruct 
the original image from the input latent vector. Notably, it 
diverges from the standard ResNet-18 by accepting a vector as 
input and employing transposed convolution layers instead of 
traditional convolution. The use of a lighter decoder is a strategic 
choice to balance the computational efficiency with the quality 
of image reconstruction, leveraging the detailed encoding while 
ensuring a lower computational burden during the decoding 
process[6]. 
To reconstruct input image at the output we calculate mean 
square error between reconstructed and original image. The loss 
function is defined as follow: 
(1) =1 − 
 Where  indicates target pixel and  refers to reconstructed 
pixel, and  is the number of pixels.  
 This novel dual-input approach, coupled with the strategic 
segmentation and fusion of data in the latent stage, positions our 
architecture as a potentially powerful and easy to implement tool 
in applications requiring detailed image analysis and 
reconstruction. 
B. Downstream Stage 
In the Downstream stage of our research, we focus on the 
application of transfer learning as an extension of our 
autoencoder-based framework. Post the successful training of 
the autoencoder for reconstructing original image patches, we 
initiate a critical transition in our methodology. This transition 
involves detaching the decoder component of the autoencoder 
while retaining the encoder part. The encoder, now functioning 
as a feature extractor, is augmented with a classification layer at 
its terminus. We feed the encoder with spatial and spectral data 
from our limited set of training samples. This process is 
meticulously designed to ensure that the encoder, now 
repurposed, effectively adapts to its new role in the pipeline. 
This repository contains the MATLAB implementation of the paper:

High-Resolution Remote Sensing Image Classification With Limited Training Data

Mehdi Ariaei, Hassan Ghassemian, and Maryam Imani

13th Iranian / 3rd International Machine Vision and Image Processing Conference (MVIP), 2024.

The proposed method leverages unsupervised feature learning with an asymmetric dual-input autoencoder followed by transfer learning on multi-spectral WorldView-3 satellite imagery. The pipeline extracts rich spatial–spectral representations using autoencoders and then re-trains the encoder on limited labeled samples to achieve accurate per-pixel classification.
