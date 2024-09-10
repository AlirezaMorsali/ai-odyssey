<p align="center">
  <img src="../../images/training_epoch_150000.png" alt="Results" width="500"/>
</p>


This section provides an implementation of the key concepts from the paper "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (Sohl-Dickstein et al., 2015), specifically focusing on diffusion probabilistic models.

# Overview

The model implemented in this section uses a diffusion process to systematically destroy data structure and then learns to reverse the process, restoring the structure in a generative manner. This approach allows for flexible probabilistic modeling, making it suitable for tasks such as density estimation, sampling, and inference.

# Key Components

* MLP Architecture
    * The MLP serves as the neural network backbone for learning the reverse diffusion process. It consists of a head network to extract hidden features from input data and multiple tail networks for step-wise diffusion reverse operations.
* Diffusion Process:
    * Forward Process
        * This process gradually adds noise to data, converting it into a simple, tractable distribution (e.g., Gaussian).
    * Reverse Process:
        * The model learns to reverse the diffusion, progressively denoising the data to regenerate samples from the original distribution.
* Loss Function:
    * The training loss consists of a KL divergence-based objective, ensuring that the reverse diffusion process reconstructs the data distribution as accurately as possible.

# Training

A sample training is provided, where the diffusion model is trained on a synthetic Swiss Roll dataset. The model is optimized using Adam optimizer, and the training progress is periodically visualized.

