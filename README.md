# Chinese Chess Piece Recognition - EECS 349 Machine Learning Final Project

## Evan Chien / Miaoding Dai
---
## <span style="color:blue">Goal</span>

The goal of our project is to classify Chinese Chess pieces. Given an image of a chess piece (traditional chinese character on top of it), we hope our classifier recognizes the color and the types of chess pieces at a strong confidence level.

## <span style="color:blue">Our model</span>
We chose Keras + TensoeFlow to build up our CNN model. The moddel has 3 layers of 32, 32, 64 feature maps, the input is color image with 56*56 size and the output is the class of the chess in the picture for sure.

## <span style="color:blue">Dataset</span>

The Chinese chess has black and red chess pieces holding by two players. Each one has 7 different kind of chess.Thus, there are 14 classes in out dataset.

The data classes in order are:

<img src="img/b_jiang.png" alt="drawing" width="100px"/><img src="img/b_ju.png" alt="drawing" width="100px"/><img src="img/b_ma.png" alt="drawing" width="100px"/><img src="img/b_pao.png" alt="drawing" width="100px"/><img src="img/b_shi.png" alt="drawing" width="100px"/><img src="img/b_xiang.png" alt="drawing" width="100px"/><img src="img/b_zu.png" alt="drawing" width="100px"/>

##
<img src="img/r_bing.png" alt="drawing" width="100px"/>
<img src="img/r_ju.png" alt="drawing" width="100px"/><img src="img/r_ma.png" alt="drawing" width="100px"/><img src="img/r_pao.png" alt="drawing" width="100px"/><img src="img/r_shi.png" alt="drawing" width="100px"/><img src="img/r_shuai.png" alt="drawing" width="100px"/><img src="img/r_xiang.png" alt="drawing" width="100px"/>


The dataset we use is manually taken by digital camera as in below.
<img src="img/setting.JPG" alt="drawing" width="400px"/>

We took 18 pictures of each chess type as the source of out training and validation data. And, with the help of  `ImageDataGenerator` in `Keras`, we were able to generate a dataset of 14,000 pictures (1000 per class) for training and 2,800 pictures (200 per class) for validation. The augmentation includes rotation, shearing, shift and zoom. You can find the augmented training/verification dataset in the data folder. Below are the snap shots of the augmented images.
<img src="img/aug_1.png" alt="drawing" width="300px"/>    <img src="img/aug_2.png" alt="drawing" width="300px"/>

As for testing, we decited to capture the frames from camera live feed for prediction.
## <span style="color:blue">Functions</span>
Dependencies: Keras, Numpy, Python 3, OpenCV, PIL, TensorFlow
For details, please refer to the comments in each file.

* <span style="text-decoration:underline">toy_cnn_mini.py</span>: The training function. It reads in images with pre-defined size (default:56). After 10 epochs, 400 steps per epoch training, a `.h5` file with model parameters is created and ready for prediction.
* <span style="text-decoration:underline">evaluate_model_spec.py</span>: The performance evaluation function. It reads in a batch of test images, converts them to the pre-defined size and output the confusion matrix and the classification report of the model.
* <span style="text-decoration:underline">rt_test.py</span>: This is the real time test function that we use for testing.
* <span style="text-decoration:underline">vgg16_cnn_bottleneck.py</span>: This is the function we use to tune the bottleneck feature.

## <span style="color:blue"> Test result</span>
The model used in the test below is `toy_cnn_mini_model_30_1800_5epo_0.97.h5`. Below are two video clips with different lighting conditions. Please note that the implementation of this function is without localization and thus we create a ROI in the center for prediction and we have to place the chess pieces near the center point.

As you can see in the links below, despite we have strong confidence in training/validation, our classifier still has some problem with class `b_ma`, `b_xiang` and `'b_pao`. These are the ones in black with lower precision. Surprisingly, the classifier does better with red chess pieces than with black ones.

[Video 1 with light condition 1](https://youtu.be/2Fv16iSG5F4)

[Video 2 with light condition 2](https://youtu.be/BOO4li_PxPQ)


## <span style="color:blue">Analysis</span>
### <span style="text-decoration:underline">Learning rate</span>
Learning rate is an important question as if we have the idea of how fast our classifier converges to its target accuracy. It also gives us the idea of how good our classifier is to the task.

Here, we tried to cut in from a different angle. That is, how much the number of our data affects the learning rate. We generated different numbers (range from 100 to 20000) of pictures out from different number of `ORIGINAL` pictures (10, 20, 30 per class) with ImageDataGenerator.

First, we evaluate when the accuracy hits 50%.
* 10 pics/class
    * Hits 50% with 2000 pictures in total
<img src="graph/10-2000.png" alt="drawing" width="600px"/>
* 20 pics/class
    * Hits 50% with 1000 pictures in total
<img src="graph/20-1000.png" alt="drawing" width="600px"/>
* 30 pics/class
    * Hits 50% with 1000 pictures in total
<img src="graph/30-1000.png" alt="drawing" width="600px"/>
<br></br>

Now, let's evaluate when they reach 90%.
* 10 pics/class
    * Hit 90% with 5000 pictures in total
<img src="graph/10-5000.png" alt="drawing" width="600px"/>
* 20 pics/class
    * Hits 90% with 5000 pictures in total
<img src="graph/20-5000.png" alt="drawing" width="600px"/>
* 30 pics/class
    * Hits 90% with 5000 pictures in total
<img src="graph/30-5000.png" alt="drawing" width="600px"/>

And, last, the learning curve of training/validation versus original sample counts. In fact, as you can see in the graphs, most of the time the differences between curves is not huge. we feel if we have larger number gap we would see the gap more easily.
<img src="img/training_loss.png" alt="drawing" width="600px"/>
<img src="img/validation_loss.png" alt="drawing" width="600px"/>



### <span style="text-decoration:underline">Confusion matrix</span>
Another important thing we need to think of is which classes in the dataset our classifier has strong confidence and which doesn't.

Below is the confusion matrix and the classification report of our fine-tuned model. The test dataset is also generated by image augmentation and has 200 samples per class.

From left to right and from top to down are in the order of this list:
`['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu', 'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']`

![matrix](img/confusion_matrix.png)
<img src="img/classification_report.png" alt="drawing" width="540px"/>

We can see that `r_ju` and `r_xiang` are with lower precision and the red ones (classes starting with r_) have lower precision comparing with the black.

## <span style="color:blue">Future Works</span>
### <span style="text-decoration:underline">Influential factors</span>
In our tests, we see mis-classifications on some of the black chess pieces and the red ones performs much better. This contradicts the classification report and we will find out whether it is the camera or other factors affecting the accuracy.
### <span style="text-decoration:underline">Optimization</span>
As mentioned in the status update, we are still working on the bottlenecks from pretrained VGG16 CNNarchitecture and will keep updating.

### <span style="text-decoration:underline">Comparing BoW</span>
Visual BoW is a technique people usually use to solve tasks like this. A study and a comparison between CNN and BoW on smaller sample is what we will work on in the near future.
