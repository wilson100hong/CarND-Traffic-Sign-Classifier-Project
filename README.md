# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/train_distribution.png 
[image2]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/valid_distribution.png
[image3]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/test_distribution.png
[image4]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/gray_scale.png
[image5]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/contrast.png
[image6]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/lenet.png

[image7]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/my-traffic-signs/11.png
[image8]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/my-traffic-signs/12.png
[image9]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/my-traffic-signs/24.png
[image10]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/my-traffic-signs/31.png
[image11]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/my-traffic-signs/38.png

[image12]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/vis_1.png
[image13]: https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_images/vis_2.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
See my [github](https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project):
* [README.md](https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)
* [output.html](https://github.com/wilson100hong/CarND-Traffic-Sign-Classifier-Project/blob/master/output.html)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set = 34799
* The size of the validation set = 4410
* The size of test set = 12630
* The shape of a traffic sign image = [32, 32, 3]
* The number of unique classes/labels in the data set = 43

#### 2. Include an exploratory visualization of the dataset.

See output.html, which includes:
* 3 sample images for each traffic signs.
* Distribution (count) for each signs in different dataset:
  * Training Set Distribution
  
  ![alt text][image1]
  * Validation Set Distribution
  
  ![alt text][image2]
  
  * Test Set Distribution
  
  ![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
* I tried to convert the images into grayscale using [perceptual luminance-preserving](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale). However, grascale images results in worst accuracy in validation and test data set, compared to RGB. So I decide **NOT** to use grayscale.

  Here are the examples for grayscaled images:
  ![alt text][image4]

* I notice some images have low contrast, which cause dark images and might affect the training. I have tried different techniques to adjust their contrast by balancing the brightness distribution:
  * [Histogram equalization](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
  * [CLAHE (Contrast Limited Adaptive Histogram Equalization)](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
  * [Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction)
  
  I decide to use Gamma correction because it results in better training accuracy. The comparison for Gamma correction before and after:
  ![alt text][image5]

* In the last step, I do normalization to convert image range [0, 255] to [0, 1]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
I use LeNet as bsical model and add additonal layers
![alt text][image6]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 28x28x6 	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU 					|												|
| Max pooling			| 2x2 stride, valid padding, outputs 28x28x16 	|
| Fully connected		| 400x120										|
| RELU 					|												|
| Dropout 				| keep prob = 0.5								|
| Fully connected		| 128x84										|
| RELU 					|												|
| Dropout 				| keep prob = 0.5								|
| Fully connected		| 84x34											|
| Softmax				| 												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* AdamOptimizer with exponential decay learning rate: Initial learning rate = 0.0009, beta1 = 0.9, beta2 = 0.999
* Batch size = 128. I have tried 100, 128 and 200 but there is not too much difference.
* Epochs = 60 to make sure the model to converge.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
My final model results were:
* training set accuracy: 1.000
* validation set accuracy: 0.965 
* test set accuracy: 0.957

If an iterative approach was chosen:
* Choose LeNet as the architecture
* Trainig accuracy is not stable, so I increase the epoch to 60. The accuracy converges better but still osciallates
* Lower traning rate to 0.0009 and the accuracy oscillation gets better.
* Use exponential deacy in Adam optimizer to further stablizae the final training accuracy with beta1 = 0.9, beta2 = 0.999.
* Notice some testing images classified wrong, which might due to overfitting, so I add dropout layers between fully connected layer and fix such issues.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are five German traffic signs picked form the total 12 images I tried::
(See output.html for more images and better visualization result)

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
Here are the results of the prediction:

| Image										| Prediction									| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection		| Right-of-way at the next intersection			| 
| Priority road								| Priority road									|
| Road narrows on the right					| Road narrows on the right						|
| Wild animals crossing						| Wild animals crossing			 				|
| Keep right								| Keep right									|


The model was able to correctly guess all the images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

* Right-of-way at the next intersection

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Right-of-way at the next intersection			| 
| .00     				| Double curve 									|
| .00					| Beware of ice/snow							|
| .00	      			| Speed limit (100km/h)					 		|
| .00				    | Pedestrians     								|

* Priority road	

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Priority road									| 
| .00     				| Traffic signals 								|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (30km/h)					 		|
| .00				    | Speed limit (50km/h)     						|


* Road narrows on the right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road narrows on the right						|
| .00     				| Bicycles crossing 							|
| .00					| Road work										|
| .00	      			| Double curve					 				|
| .00				    | Dangerous curve to the left    				|

* Wild animals crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Wild animals crossing							| 
| .00     				| Road work 									|
| .00					| Slippery road									|
| .00	      			| No passing for vehicles over 3.5 metric tons	|
| .00				    | Bicycles crossing    							|


* Keep right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep right									| 
| .00     				| Turn left ahead 								|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (30km/h)					 		|
| .00				    | Speed limit (50km/h)    						|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
* First layer

  Recognize the contour (ciricle), and the inner stipe

![alt text][image12]

* Second layer


![alt text][image13]

