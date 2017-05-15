# Udacity Nanodegree Foundation Program

This is the repository of all the projects I will carry out during [Udacity Deep Learning Nanodegree Foundation Program]("https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101")


## Project 1 -- your first neural network

Using only numpy, we build a neural network from scratch to carry out predictions on daily bike rental ridership.

In this project, you'll get to build a neural network from scratch to carry out a prediction problem on a real dataset! By building a neural network from the ground up, you'll have a much better understanding of gradient descent, backpropagation, and other concepts that are important to know before we move to higher level tools such as Tensorflow. You'll also get to see how to apply these networks to solve real prediction problems!

```
conda create --name dlnd python=3
source activate dlnd
conda install numpy matplotlib pandas jupyter notebook
jupyter notebook DLND\ Your\ first\ neural\ network.ipynb
```


## Project 2 -- Image Classification 

Using TensorFlow, the objectif is to classify Images from [CIFRA-10 dataset]("https://www.cs.toronto.edu/~kriz/cifar.html").

Here is the different steps contained in "the dlnd_image_classification" notebook :
- Get and Explore the data from [CIFRA-10 dataset]("https://www.cs.toronto.edu/~kriz/cifar.html") : The dataset consists of airplanes, automobiles, birds, cats, deers, dogs, frogs, horses, ships end trucks.
- Pre-process functions : Normalizing, One-Hot encoding, No need to randomize the images (the dataset is already randomized)
Normalize the images : 2 solutions using np.linalg.norm => compute the norm of the matrix or divide by 255.

- Build a convolutional layer :
  - **Input** : Implement 3 tensors for the images, the labels and the drop out (neural_net_image_input, neural_net_label_input, neural_net_keep_prob_input)
  - **Apply convolution and max pooling** using `tf.nn.conv2d`, `tf.nn.bias_add`, `tf.nn.relu` and `tf.nn.max_pool`.
  - **Implement the flatten function** to change the dimension of x_tensor from a 4-D tensor to a 2-D tensor using `tf.reshape`.
  - **Then fully connected layer** to x_tensor with the shape (Batch Size, num_outputs). XW + b and apply relu function.
  - **The output layer** : apply a fully connected layer to x_tensor with the shape (Batch Size, num_outputs) without applying activation functions.

**My Convolutional Model**
  - Apply 1, 2, or 3 Convolution and Max Pool layers
  - Apply a Flatten Layer
  - Apply 1, 2, or 3 Fully Connected Layers and Apply TensorFlow's Dropout using keep_prob
  - Apply an Output Layer
  - Return the output

**My Parameters**
- Normalize using 255 (np.linalg.norm)
- conv2d_maxpool => weights stddev = 0.1
- fully_conn => weights stddev = 0.1
- output => weights stddev = 0.1
- conv_net => Play around with different number of outputs, kernel size and stride
- Hyperparameters :
  - epochs = 20
  - batch_size = 128
  - keep_probability = 0.85
