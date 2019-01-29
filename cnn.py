import keras
import numpy as np

import matplotlib.pyplot as plt
from keras import optimizers, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import sys


from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from mnistloader.mnist_loader import MNIST

old_stdout = sys.stdout
log_file = open("summaryRFC.log","w")
sys.stdout = log_file

#
# Parse the Arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--save_model", type=int, default=-1)
# ap.add_argument("-l", "--load_model", type=int, default=-1)
# ap.add_argument("-w", "--save_weights", type=str)
# args = vars(ap.parse_args())

# # Read/Download MNIST Dataset
# print('Loading MNIST Dataset...')
# dataset = fetch_mldata('MNIST Original')
#
# # Read the MNIST data as array of 784 pixels and convert to 28x28 image matrix
# mnist_data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
# mnist_data = mnist_data[:, np.newaxis, :, :]
#


# Load MNIST Data
print('\nLoading MNIST Data...')
data = MNIST('./mnistloader/dataset/')

print('\nLoading Training Data...')
img_tra, labels_tra = data.load_training()
tra_img = np.array(img_tra)
tra_labels = np.array(labels_tra)

print('\nLoading Testing Data...')
img_te, labels_te = data.load_testing()
te_img = np.array(img_te)
te_labels = np.array(labels_te)


#Features
X = tra_img


#Labels
y = tra_labels



b = np.reshape(X,47040000)
X = (np.reshape(b, (60000, 28, 28,1))).astype('float32')
print('X_train shape:', X.shape) #X_train shape: (60000, 28, 28, 1)
#mnist_data / 255.0, dataset.target.astype("int")


# Divide data into testing and training sets.
train_img, test_img, train_labels, test_labels = train_test_split(X,y,test_size=0.1)


#reshaping
#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
# if k.image_data_format() == 'channels_first':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# #more reshaping
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255



# image_index = 7777 # You may select anything up to 60,000
# a = train_img[image_index]
#
# print(train_img[image_index]) # The label is 8
# plt.imshow(two_d, cmap='Greys')
#
# train_img, test_img, train_labels, test_labels = train_test_split(mnist_data / 255.0, dataset.target.astype("int"))

# Now each image rows and columns are of 28x28 matrix type.
img_rows, img_columns = 28, 28

# Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
total_classes = 10  # 0 to 9 labels
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)
input_shape = (img_rows, img_columns, 1)

model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(Flatten())
#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :)
model.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(Dense(10, activation='softmax'))
# Defing and compile the SGD optimizer and CNN model
print('\n Compiling model...')
#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 10
#model training
model_log = model.fit(train_img, train_labels,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(test_img, test_labels))
#Save the model
# serialize model to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save_weights("model_digit.h5")
print("Saved model to disk")

score = model.evaluate(test_img, test_labels, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904

sys.stdout = old_stdout
log_file.close()

import os
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# clf = CNN.build(width=28, height=28, depth=1, total_classes=10,
#                 Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None)
# clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Initially train and test the model; If weight saved already, load the weights using arguments.
# b_size = 128  # Batch size
# num_epoch = 20  # Number of epochs
# verb = 1  # Verbose

# # If weights saved and argument load_model; Load the pre-trained model.
# if args["load_model"] < 0:
#     print('\nTraining the Model...')
#     clf.fit(train_img[None,:], train_labels[None,:], batch_size=b_size, epochs=num_epoch, verbose=verb)
#
#     # Evaluate accuracy and loss function of test data
#     print('Evaluating Accuracy and Loss Function...')
#     loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
#     print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))
#
# # Save the pre-trained model.
# if args["save_model"] > 0:
#     print('Saving weights to file...')
#     clf.save_weights(args["save_weights"], overwrite=True)
#
# # Show the images using OpenCV and making random selections.
# for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
#     # Predict the label of digit using CNN.
#     probs = clf.predict(test_img[np.newaxis, num])
#     prediction = probs.argmax(axis=1)
#
#     # Resize the Image to 100x100 from 28x28 for better view.
#     image = (test_img[num][0] * 255).astype("uint8")
#     image = cv2.merge([image] * 3)
#     image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
#     cv2.putText(image, str(prediction[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#
#     # Show and print the Actual Image and Predicted Label Value
#     print('Predicted Label: {}, Actual Value: {}'.format(prediction[0], np.argmax(test_labels[num])))
#     cv2.imshow('Digits', image)
# cv2.waitKey(0)