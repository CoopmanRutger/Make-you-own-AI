# Make-you-own-AI
Tutorial - make your own AI 

## Folder structure:

neuralNetworkTrain.py (this is where you train the neural network)
neuralNetworkTest.py (this is where you run tests)
neuralNetwerkModel.h5 (this is the model you trained)
Test(Folder)
  - 0 (Folder)
    - Pictures of 1 category (e.g. the board)
  - 1 (Folder)
    - Pictures of 1 category (e.g. red discs)
  - 2 (Folder)
    - Pictures of 1 category (e.g. yellow discs)
Train(Folder)
  - 0 (Folder)
    - Pictures of 1 category (This must match Test Folder)
  - 1 (Folder)
    - Pictures of 1 category
  - 2 (Folder)
    - Pictures of 1 category
    
    
## Steps

### Step 1:
Organize your photos according to the folder structure. (In my case a board, red and yellow discs.) All the same size and colour in either colour or black and white. If you have more categories, add a new folder. You divide the photos by category 80/20 (train/test).



### Step 2 (neuralNetworkTrain.py):
First some necessary imports (Line 1-8).

I have chosen to take pictures of width:100pixels, height:100pixels and depth:1 (which means greyscale picture). So, these are black and white pictures of 100 by 100 pixels. I also reference to training and test folder. (Line 9-21)


### Step 3 (neuralNetworkTrain.py):
Normalize your photo generator (Lines 23-24). I use TensorFlow and Keras because after research I found this seemed the best.


### Step 4 (neuralNetworkTrain.py):
Setting up your generators. Both for training and testing.  For reading your photo batch by batch. See line 27-40


### Step 5 (neuralNetworkTrain.py):
If you want to know more about this, I would like to refer you to a detailed explanation on the Keras site (see above).
Setting up the neural network (Lines 57-69), I have chosen a Sequential model from Keras, that is enough for the network. I have chosen five hidden layers, the first three-layer I chose the Conv2D layer followed by a MaxPool2D layer. Conv2D layer creates a convolution kernel from 5 on 5 that is convolved with the layer input to produce a tensor of outputs. If use_bias is true, a bias vector is created and added to the outputs. Finally, if activation is not none, it applies to the outputs as well. The max-pooling operation for territorial data. Layer one, two, three and five using a Rectified Linear Unit (RELU) as activation function (Applies an activation function to an output).

The fourth layer (Lines 72-75) is a Flatten layer. Flattens the input, does not affect the batch size and as the last hidden layer uses a Dense with RELU. Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, the kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).

For my output layer (Lines 78-79), I chose Dense again, this time with SoftMax. This one is better for categorizing than e.g. a RELU.


### Step 6 (neuralNetworkTrain.py):

Compiling the network (Line 82-83). I have chosen ‘categorical_crossentropy’ because it is about multiple categories. If there were, only two you could still use binary. As optimizer I have chosen ‘sgd’, I noticed little difference with another one and as metrics ‘accuracy’.

### Step 7 (neuralNetworkTrain.py):
List the entire network (line 86). This gives you a nice detailed overview of what your network looks like, with the different layers, params and nodes. How many you can train and how many you cannot, in this case, they are all trainable (85 403 params).


### Step 8 (neuralNetworkTrain.py):
Training your model (lines 91-98). You can choose different settings such as: What do you want to train? My train data of course. How many steps per epoch do you want? 26. How many epochs do you want? 2. Do you want to validate and what data do you want to use for this and how many steps do you want to validate? Yes, my testing data and every 3 steps.

You have to play with this according to the number of pictures and their purpose. With each epoch, you get four digits. Loss and Val_loss (as low as possible) and accuracy and val_accuracy (as near to 1 as possible). Too much training is not good either. Then you may get worse results. Val stands for validation. They make a difference between training data and validation data. You can see that the accuracy of training data increases while that of test data does not change outside into 3, this can be by chance.

An example:

Epoch 1/5: loss: 0.9774 - accuracy: 0.5814 - val_loss: 1.1762 - val_accuracy: 0.6667
Epoch 2/5: loss: 0.9016 - accuracy: 0.6124 - val_loss: 0.4081 - val_accuracy: 0.6667
Epoch 3/5: loss: 0.7712 - accuracy: 0.7308 - val_loss: 0.7810 - val_accuracy: 0.9333
Epoch 4/5: loss: 0.5668 - accuracy: 0.7969 - val_loss: 0.4121 - val_accuracy: 0.6667
Epoch 5/5: loss: 0.5076 - accuracy: 0.8217 - val_loss: 0.5526 - val_accuracy: 0.6667
Afterwards, you can save the model (line 102) and the weights if you want.


### Step 9 (neuralNetworkTest.py):
Testing network usage in another file (see neuralNetworkTest.py). You can do this in the same file but then you can only test once, or you have to put things in comments.

For testing you need of course a picture, read it with cv2 or PIL of your choice. Convert the 3D-array to a NumPy 3D-array. Load your trained model with ‘load_model’ and you normalize it as if you did your other pictures and change it to a 4D-array. Because it only accept tensors that is a 4D-array. Finally, you do a 'predict_classes' of the fourD-array based on your model and then you get the result.


### Step 10 (neuralNetworkTrain.py):
Optimize your network!
  - Add pictures
  - Change layers
  - Add layers
  - Change epochs
  - …
