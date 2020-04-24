# Autoencoder SearchEngine

Search Engine using Autoencoder.

Description:

Use Stacked Autoencoders to learn low dimensional rich hidden representation of
different type of images.

▪ Addition of noise in the encoding layer to force hidden representation to be either 0 or 1 and to help in learning the features from the image. 

▪ Images part of repository to be searched for, are encoded in a hidden representation of 0’s and 1’s
and stored in a matrix.

▪ Query image is passed to the autoencoder to extract it’s hidden representation,which is XORed with
the repository matrix to get a score of similarity of the hidden representation with each of the image
representation in the matrix


Training error and the closest matchable image found through XOR between the query image representation and each image hidden representation in the repository matrix to get a score of similarity.

Saving the model after every 1000th iteration, stopped when validation loss is around 0.50

![autoencoder_loss](https://user-images.githubusercontent.com/23450113/80243698-f8ac0d00-8667-11ea-94ee-a0beb4829c19.png)


The closest matchable image 
The bottom image represent the image that we are searching for in the matrix

The top 3 images represents the closest matched images from the repository matrix(with the rightmost having the highest score)
![best1](https://user-images.githubusercontent.com/23450113/80243631-dd410200-8667-11ea-99c0-5bc832c83327.png)
![best_1](https://user-images.githubusercontent.com/23450113/80243637-df0ac580-8667-11ea-8808-739b18ddc17c.png)

