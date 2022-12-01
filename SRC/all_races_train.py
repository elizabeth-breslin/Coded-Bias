import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import os
import csv
import shutil
import pandas as pandasForSortingCSV
from random import shuffle
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA"))

#data_list = os.listdir("C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace")


### This Code is to split the data set with all races randomly into a train data set and a test data set ###
all_female = os.listdir('C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Female')
all_male = os.listdir('C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Male')
white_female = os.listdir('C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Female')
white_male = os.listdir('C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Male')
random.shuffle(all_female)
random.shuffle(all_male)
random.shuffle(white_female)
random.shuffle(white_male)
white_female_data = white_female[:1000]
white_male_data = white_male[:1000]

train_all_female_data = all_female[100:200]
test_all_female_data = all_female[:100]
train_all_male_data = all_male[100:200]
test_all_male_data = all_male[:100]

# for items in train_all_female_data:
#    original = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Female/'+items
#    target = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/TRAIN/'+items
#    shutil.move(original, target)

#for items in test_all_female_data:
#    original = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Female/'+items
#    target = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/TEST/'+items
#   shutil.move(original, target)

#for items in train_all_male_data:
#     original = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Male/'+items
#     target = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/TRAIN/'+items
#     shutil.move(original, target)

# for items in white_male_data:
#     original = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Male/'+items
#     target = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/TRAIN/'+items
#     shutil.move(original, target)

# for items in white_female_data:
#     original = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Female/'+items
#     target = r'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/TRAIN/'+items
#     shutil.move(original, target)

train_white_female = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Female/TRAIN'
train_white_male = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/White - Male/TRAIN'
train_all_female = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Female/TRAIN'
train_all_male = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Male/TRAIN'
test_all_female = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Female/TEST'
test_all_male = 'C:/Users/Student/Downloads/UVA/Fourth Year/Project/Project 3/PHOTOS/UTKFace/DATA/All races - Male/TEST'
image_size = 128


def train_data_all():
    train_data_all_female = []
    train_data_all_male = []
    for image1 in tqdm(os.listdir(train_all_female)):
        path = os.path.join(train_all_female, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (image_size, image_size))
        train_data_all_female.append(img1)
    for image2 in tqdm(os.listdir(train_all_male)):
        path = os.path.join(train_all_male, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (image_size, image_size))
        train_data_all_male.append(img2)

    train_data_all = np.concatenate((np.asarray(train_data_all_female), np.asarray(train_data_all_male)), axis=0)
    return train_data_all


def test_data():
    test_data_all_female = []
    test_data_all_male = []
    for image1 in tqdm(os.listdir(test_all_female)):
        path = os.path.join(test_all_female, image1)
        img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.resize(img1, (image_size, image_size))
        test_data_all_female.append(img1)
    for image2 in tqdm(os.listdir(test_all_male)):
        path = os.path.join(test_all_male, image2)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (image_size, image_size))
        test_data_all_male.append(img2)

    test_data = np.concatenate((np.asarray(test_data_all_female), np.asarray(test_data_all_male)), axis=0)
    return test_data


train_data_all = train_data_all()
test_data = test_data()

x_data = np.concatenate((train_data_all,test_data),axis=0)
x_data = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


z1 = np.zeros(100)
o1 = np.ones(100)
Y_train = np.concatenate((o1, z1), axis=0)
z = np.zeros(100)
o = np.ones(100)
Y_test = np.concatenate((o, z), axis=0)

y_data=np.concatenate((Y_train,Y_test),axis=0).reshape(x_data.shape[0],1)
print("X shape: " , x_data.shape)
print("Y shape: " , y_data.shape)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]

x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])

print("X train flatten",x_train_flatten.shape)
print("X test flatten",x_test_flatten.shape)
#
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_test = y_test.T
y_train = y_train.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
#

#LOGISTIC REGRESSION
def initialize_weights_and_bias(dimension):
     w = np.full((dimension,1),0.01)
     b = 0.0
     return w, b

#
def sigmoid(z):
     y_head = 1/(1+np.exp(-z))
     return y_head


def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

#
def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
    global gradients
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iterarion):

        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))

    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#
# #
def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction
#
#
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100, 2)))
    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100, 2)))

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
