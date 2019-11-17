# HK Santosh - EIP 4 - Week 1

## CNN Result/Score: [0.04768682908062915, 0.9901]
Score of more than 99 was achieved by using additional convolution layers. 
1st_DNN_tf2.ipynb is the notebook which contains improved results and also has changes to adapt to Tensorflow 2
1st_DNN.ipynb (ignore this for now) - This has improved results but uses Tensorflow 1.x and is not tuned to exceed the score of 99


## Definitions
1. Convolution: An operation that transforms input data into a specific feature(s) rich output using a filter/kernel.
2. Filter/Kernel: An operator that allows only the desired qualities of the input to pass through it.
3. Epochs: One full pass/iteration of all training samples through a neural network.
4. 1x1 Convolution: Operation that extracts a specific feature from the input data and generates output of same dimension as input.
5. 3x3 Convolution: Operation that extracts specific feature(s) from input data by using a 3x3 filter and generates output of lesser dimension (by 2x2).
6. Feature Maps: A map of desired qualities/features. Typically an input/output to/of convolution.
7. Activation Function: A function designed to activate a neuron if it's input is useful for achieving the model's objective.
8. Receptive Field: During convolution, at a time, filter is applied only to a part of input data. This part is called as receptive field.
