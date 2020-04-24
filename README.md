# SearchEngine

Training error and the closest matchable image found through XOR between the query image representation and each image hidden representation in the repository matrix to get a score of similarity.

Saving the model after every 1000th iteration, stopped when validation loss is around 0.50

![autoencoder_loss](https://user-images.githubusercontent.com/23450113/80243698-f8ac0d00-8667-11ea-94ee-a0beb4829c19.png)


The closest matchable image 
The bottom image represent the image that we are searching for in the matrix

The top 3 images represents the closest matched images from the repository matrix(with the rightmost having the highest score)
![best1](https://user-images.githubusercontent.com/23450113/80243631-dd410200-8667-11ea-99c0-5bc832c83327.png)
![best_1](https://user-images.githubusercontent.com/23450113/80243637-df0ac580-8667-11ea-8808-739b18ddc17c.png)

