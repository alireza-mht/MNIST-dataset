
import sys
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from mnistloader.mnist_loader import MNIST
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

old_stdout = sys.stdout
log_file = open("ABsummary.log","w")
sys.stdout = log_file

print('\nLoading MNIST Data...')

data = MNIST('./mnistloader/dataset/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels

print('\nPreparing Classifier Training and Validation Data...')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)


print('\nAdaboost Classifier with n_estimators = 100')
print('\nPickling the Classifier for Future Use...')
clf = AdaBoostClassifier(n_estimators=110)
clf.fit(X_train,y_train)

with open('MNIST_AdaBoost.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_AdaBoost.pickle','rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
confidence = clf.score(X_test,y_test)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_test)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_test, y_pred)

print('\nCalculating precision score of Predictions...')
precision = precision_score(y_test, y_pred,average='macro')

print('\nCalculating recall score of Predictions...')
recall = recall_score(y_test, y_pred,average='macro')

print('\nCalculating F1 of Predictions...')
f1 = f1_score(y_test, y_pred,average='macro')

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_test,y_pred)

print('\nAdaboost Trained Classifier Confidence: ',confidence)
print('\nPredicted Values: ',y_pred)
print('\nAccuracy of Classifier on Validation Image Data: ',accuracy)
print('\nPrecision of Classifier on Validation Images: ',precision)
print('\nRecall of Classifier on Validation Images: ',recall)
print('\nF1 of Classifier on Validation Images: ',f1)
print('\nConfusion Matrix: \n',conf_mat)



# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


print('\nMaking Predictions on Test Input Images...')
test_labels_pred = clf.predict(test_img)

print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
acc = accuracy_score(test_labels,test_labels_pred)

print('\nCalculating precision score of Trained Classifier on Test Data...')
tprecision = precision_score(test_labels,test_labels_pred,average='macro')

print('\nCalculating recall score of Trained Classifier on Test Data...')
trecall = recall_score(test_labels,test_labels_pred,average='macro')

print('\nCalculating F1 of Trained Classifier on Test Data...')
tf1 = f1_score(test_labels,test_labels_pred,average='macro')



print('\n Creating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

print('\nPredicted Labels for Test Images: ',test_labels_pred)
print('\nAccuracy of Classifier on Test Images: ',acc)
print('\nPrecision of Classifier on Test Images: ',tprecision)
print('\nRecall of Classifier on Test Images: ',trecall)
print('\nF1 of Classifier on Test Images: ',tf1)
print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.show()

sys.stdout = old_stdout
log_file.close()
