import keras
import numpy as np

import matplotlib.pyplot as plt
from keras import optimizers, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, recall_score, precision_score
from keras.models import model_from_json
import sys

from keras import metrics
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from mnistloader.mnist_loader import MNIST

old_stdout = sys.stdout
log_file = open("summarycnn.log","w")
sys.stdout = log_file

print("cnn Log :\n\n")


# Load MNIST Data
data = MNIST('./mnistloader/dataset/')

img_tra, labels_tra = data.load_training()
tra_img = np.array(img_tra)
tra_labels = np.array(labels_tra)

img_te, labels_te = data.load_testing()
te_img = np.array(img_te)
te_labels = np.array(labels_te)


#Features
X = tra_img
X_test = te_img

#Labels
y = tra_labels
y_test = te_labels


b = np.reshape(X,47040000)
X = (np.reshape(b, (60000, 28, 28,1))).astype('float32')

s = np.reshape(X_test,7840000)
X_test = (np.reshape(s, (10000, 28, 28,1))).astype('float32')

# Now each image rows and columns are of 28x28 matrix type.
img_rows, img_columns = 28, 28
#
# # Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
# total_classes = 10  # 0 to 9 labels
y = np_utils.to_categorical(y, 10)
y_test = np_utils.to_categorical(y_test, 10)
input_shape = (img_rows, img_columns, 1)
    #
    # model = Sequential()
    # #convolutional layer with rectified linear unit activation
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # #32 convolution filters used each of size 3x3
    # #again
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # #64 convolution filters used each of size 3x3
    # #choose the best features via pooling
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #randomly turn neurons on and off to improve convergence
    # model.add(Dropout(0.25))
    # #flatten since too many dimensions, we only want a classification output
    # model.add(Flatten())
    # #fully connected to get all relevant data
    # model.add(Dense(128, activation='relu'))
    # #one more dropout for convergence' sake :)
    # model.add(Dropout(0.5))
    # #output a softmax to squash the matrix into output probabilities
    # model.add(Dense(10, activation='softmax'))
    # # Defing and compile the SGD optimizer and CNN model
    # print('\n Compiling model...')
    # #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
    # #categorical ce since we have multiple classes (10)
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])

    # batch_size = 128
    # num_epoch = 10
    # #model training
    # model_log = model.fit(X, y,
    #           batch_size=batch_size,
    #           epochs=num_epoch,
    #           verbose=1,
    #           validation_data=(X_test, y_test))
    #Save the model
    #serialize model to JSON
    # model_digit_json = model.to_json()
    # with open("model_digit.json", "w") as json_file:
    #     json_file.write(model_digit_json)
    # # serialize weights to HDF5
    # model.save_weights("model_digit.h5")
    # print("Saved model to disk")
    #

# Model reconstruction from JSON file
with open('model_digit.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_digit.h5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


#test data accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904

#train data accuracy
score = model.evaluate(X, y, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904



    # a = model.predict(X)

    # y_pred = model.predict(X)
    # accuracy = accuracy_score(y, y_pred)
    # precision = precision_score(y, y_pred,average='macro')
    # recall = recall_score(y, y_pred,average='macro')
    # f1 = f1_score(y, y_pred,average='macro')
    # conf_mat = confusion_matrix(y,y_pred)
    #
    # print('\nSVM Trained Classifier Accuracy: ',score[1])
    # print('\nAccuracy of Classifier on Training Images: ',accuracy)
    # print('\nPrecision of Classifier on Training Images: ',precision)
    # print('\nRecall of Classifier on Training Images: ',recall)
    # print('\nF1 of Classifier on Training Images: ',f1)
    # print('\nConfusion Matrix: \n',conf_mat)
    #

    # test_labels_pred = model.predict(X_test)
    # a = test_labels_pred[:,9]
    # b = y_test[:,9]
    # Y_test = np_utils.to_categorical(a,0)
    # h_test = np_utils.to_categorical(b,0)
    #
    # acc = accuracy_score(h_test,Y_test)
    # tprecision = precision_score(Y_test,h_test,average='macro')
    # trecall = recall_score(h_test,Y_test,average='macro')
    # tf1 = f1_score(h_test,Y_test,average='macro')
    #
    # print('\nPrecision of Classifier on Test Images: ',tprecision)
    # print('\nRecall of Classifier on Test Images: ',trecall)
    # print('\nF1 of Classifier on Test Images: ',tf1)
