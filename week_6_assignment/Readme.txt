Denoising Autoencoder for MNIST

A Convolutional Denoising Autoencoder built with TensorFlow/Keras that learns to remove Gaussian noise from handwritten digit images and reconstruct clean versions, trained and evaluated on the MNIST dataset.

Overview

This project implements an end-to-end pipeline that:

1. Loads and preprocesses the MNIST handwritten digit dataset.
2. Adds artificial Gaussian noise to create noisy input images.
3. Builds and trains a Convolutional Denoising Autoencoder using noisy images as input and clean images as targets.
4. Generates denoised outputs on the test set.
5. Evaluates reconstruction quality using MSE and PSNR, with visual comparisons of original, noisy, and denoised images.

Repository Structure

Denoising-Autoencoder-App/
- notebook.ipynb              Main notebook: data prep, training, evaluation
- denoising_autoencoder.keras Trained model (saved after training)
- requirements.txt            Python dependencies
- README.md                   Project documentation

Dataset

The model is trained on the MNIST Handwritten Digit Dataset, consisting of 60,000 training images and 10,000 testing images of digits 0-9, each of size 28 x 28 pixels (grayscale).

Methodology

1. Data Preparation - Images are loaded, normalized to the range [0, 1], and reshaped to (28, 28, 1).
2. Noise Generation - Gaussian noise is added to clean images to simulate corrupted input, with pixel values clipped back to [0, 1].
3. Model Architecture - A convolutional encoder-decoder network:
   - Encoder: Conv2D layers followed by MaxPooling2D layers that progressively compress the image into a compact latent representation.
   - Decoder: Conv2D layers followed by UpSampling2D layers that reconstruct the image back to its original 28 x 28 x 1 resolution.
4. Training - The model is trained using the Adam optimizer and binary cross-entropy loss, with noisy images as input and clean images as the reconstruction target. EarlyStopping is used to monitor validation loss and prevent overfitting.
5. Evaluation - The trained model is evaluated on the test set using:
   - MSE (Mean Squared Error) - measures the average pixel-wise reconstruction error.
   - PSNR (Peak Signal-to-Noise Ratio) - measures reconstruction quality in decibels (higher is better).

Results

Metric | Noisy Images | Denoised Images
MSE    | 0.1156       | 0.0102
PSNR   | 9.37 dB      | 19.89 dB

- Training loss decreased from 0.1603 to 0.0939, and validation loss decreased correspondingly from 0.1164 to 0.0937, with both curves flattening together - indicating stable learning without significant overfitting.
- The denoising step reduced reconstruction error (MSE) by roughly 91% and improved PSNR by about 10.5 dB compared to the noisy input.
- Digits with simpler strokes (e.g. 1, 7) were reconstructed more cleanly, while digits with denser curves (e.g. 5, 8) occasionally showed slightly blurred edges, since fine structural detail is harder to recover once heavily corrupted by noise.

Key Observations

- The autoencoder successfully learns to separate noise from genuine digit structure rather than simply smoothing the entire image.
- Choosing an appropriate noise factor is important: too little noise makes the task trivial, while excessive noise (close to 1.0) starts removing real digit strokes along with the noise.
- Quantitative metrics (MSE, PSNR) combined with visual comparison give a more complete picture of model performance than either alone.

Setup and Usage

1. Clone the repository:
   git clone <repository-url>
   cd Denoising-Autoencoder-App

2. Install dependencies:
   pip install -r requirements.txt

3. Open and run the notebook:
   jupyter notebook notebook.ipynb

4. Running all cells will preprocess the data, train the autoencoder, generate denoised outputs, and save the trained model as denoising_autoencoder.keras.

Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-image

Future Work

- Experiment with different noise types and noise levels (e.g. salt-and-pepper noise, varying Gaussian variance).
- Try deeper or alternative architectures (e.g. U-Net style skip connections) to better preserve fine digit detail.
- Extend the trained model into an interactive web application for real-time denoising of user-uploaded images.

Author

Developed by Gurudeep Soni
