# Quick development of Image Recognition model

- This project aims to help accelerate the development of image recognition models in a quick manner.

- All you have to know is that the preprocessing code is present in the `prepare_data.py` file if you are not bothered about the colour of the training images and only want to focus on the training then _don't change the code here_

###### How to use?

- Use the `train_model.py` file to train your model. But first you should modify the path parameters as per the dataset you are using.

- Place your dataset in the dataset folder. Then modify the parameters section in the `train_model.py`

- The various parameters that you would want to set would be

> `num_classes = Number of classes you want to classify eg. 29`
> `no_of_epochs = Required number of epochs or iterations you want. eg. 10`
> `size = A list of the width and height of the image for eg. [200, 200]`
> `batch_size = Here you define the number of images to be fed. eg. 32`

- More adjustments can be done to the section of:-

> `Network Parameters`
> `Optimization Parameters`

> ###### Note: The Network Parameters are dependent on the model you have defined in the `cnn_model.py` file be careful in defining the new params.

#### `cnn_model.py`

- Here you will define the Network architecture that you want.

- Based on the network that you have defined remember to make changes to the network parameters in the `train_model.py` file.

#### `detector.py`

- This file is still under development as it is the file that will carry out the detection part (the final stage) of the whole project.

- It contains hardcoded list of labels and preprocessing which will soon be converted to reading from `csv` and making a separate funciton for preprocessing the image(frame) captured from the video camera.
