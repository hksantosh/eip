# MNIST Classification using less than 20k parameters and without using FC layers
* In this model, 2d batch normalization and 1D dropouts have been applied for each layer except the final layer
* First block starts with a convolution layer that is made of 10 3x3 convolution filters. The convolution layer accepts images in batches of 64 and generates 10 channels and maintains the input size 28x28 (using padding during convlution). This layer is followed by 2x2 maxpooling which reduces the dimensions of the 10 channels to 14x14. The receptive field of this block is 6x6.
* Second block consists of two layers each containing 16 3x3 convolution filters. These convolution layers increase the depth of feature maps to 16. These convolution layers are followed by 2x2 maxpooling which reduces the dimensions of feature maps to 7x7. Receptive field size has now increased to 20.
* Third block consists of a single layer containing 20 3x3 convolution filters. These convolution layers increase the depth of feature maps to 20. Receptive field size has now increased to 22 which means the input image is still not entirely visible to the model from this layer.
* Fourth block consists of two layers each containing 24 3x3 convolution filters. These convolution layers increase the depth of feature maps to 24 and also reduce the dimensions of feature maps to 3x3. Receptive field size is now 26.
* The final layer, also the output layer, contains 10 3x3 convolution filters which reduce the dimension to the desired 1x1 size. These 10 elements effectively represents the 10 digits 0-9 which we are trying to predict. log_softmax activation function is used to squash the output vectors into a value that lies between 0 to 1. We can "interpret" these output value as the probabilities of the input image belonging to the corresponding class.

## Observation
The model consistently returns accuracy of over 99.4 within 20 epochs. At times, the accuracy returned is above 99.5. Highest observed accuracy from various experiments stands at 99.55

## Stats
* **No. of parameters** in the model: **18,718**
* Batch size: 64
* Optimizer: SGD
* Learning Rate: 0.095
* Momentum: 0.95
* **Best Epoch: #19**, average loss: 0.0159, **Accuracy: 99.55%**
