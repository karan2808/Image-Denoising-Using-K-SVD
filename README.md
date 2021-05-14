# Image-Denoising-Using-K-SVD

This is a Python implementation of Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries. We add zero mean gaussion noise to an image and remove it using iterative Orthogonal Matching Pursuit over coefficients and K-SVD over Learned Dictionaries. We evaluate the performance of this method on various datasets using PSNR metric. 

## ⚙️ Setup

Create a fresh eviornment using [Anaconda](https://www.anaconda.com/download/) distribution. You can then install the dependencies with:
```shell
conda install opencv=4.1.2
conda install matplotlib
pip install scikit-learn
```
Recommended python version: 3.7

## 💾 Images

We have provided some example images in the example_images folder. More images can be downloaded using,
```shell
gdown https://drive.google.com/uc?id=1vlp9e-KS0-Z6vdm6lalX7Y_rEIgbDULQ
unzip images.zip
```

## 🖼️ Denoising the images
You can read an image, add noise to it and denoise it using the K-SVD algorithm by running,
```shell
python main.py
```
This function will also plot the original image, noisy image and denoised image. The hyper-parameters can be changed in the same file. 

## 📊 Evaluation
After denoising the image, you can compute various evaluation metrics using the original and denoised images to check the performance of the K-SVD algorithm. This can be done using,
```
python evaluate.py
```
which should print the results of the metrics. 

## References

1.  **Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries,**
    <br />
    M. Elad and M. Aharon. <br />
    [[link]](https://www.egr.msu.edu/~aviyente/elad06.pdf). IEEE Transactions on Image Processing Dec. 2006.