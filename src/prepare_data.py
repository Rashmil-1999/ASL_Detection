import os
import cv2
import numpy
from itertools import repeat
import importlib
import glob


def listALLFiles(path):
    results = glob.glob(path + "/*.jpg")
    return results


def preProcessImage(path, img_width, img_height):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32")
    img /= 255
    return img


def prepareData(size, path, num_classes):
    input_samples = []
    output_labels = []
    # for each class get all the images and apply preprocessing to each of them
    # and append them in the output and input arrays respectively.
    for _class in range(num_classes):
        path_temp = path + str(_class)
        length = len(os.listdir(path_temp))
        samples = numpy.array(
            list(
                map(
                    preProcessImage,
                    listALLFiles(path_temp),
                    repeat(size[0], length),
                    repeat(size[1], length),
                )
            )
        )
        input_samples.append(samples)
        output_labels.append(numpy.array([_class] * len(samples)))
    inputs = numpy.concatenate(input_samples, axis=0)
    outputs = numpy.concatenate(output_labels, axis=0)

    # convert to hot vectors
    output_hot_vectors = numpy.zeros((len(outputs), num_classes))
    output_hot_vectors[numpy.arange(len(outputs)), outputs] = 1
    outputs = output_hot_vectors

    # shuffle theinputs and outputs the same way
    p = numpy.random.permutation(len(inputs))
    inputs = inputs[p]
    outputs = outputs[p]

    return inputs, outputs

