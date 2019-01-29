
import sys
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from mnistloader.mnist_loader import MNIST
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

old_stdout = sys.stdout
log_file = open("RFsummary.log","w")
sys.stdout = log_file
print("Random Forest Log :\n\n")

data = MNIST('./mnistloader/dataset/')

img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)

X = train_img
y = train_labels

clf = RandomForestClassifier(n_estimators=100, n_jobs=10)
clf.fit(X,y)

with open('MNIST_RFC.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_RFC.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(X,y)
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred,average='macro')
recall = recall_score(y, y_pred,average='macro')
f1 = f1_score(y, y_pred,average='macro')
conf_mat = confusion_matrix(y,y_pred)

print('\nRFC Trained Classifier Confidence: ',confidence)
print('\nAccuracy of Classifier on Training Image Data: ',accuracy)
print('\nPrecision of Classifier on Training Images: ',precision)
print('\nRecall of Classifier on Training Images: ',recall)
print('\nF1 of Classifier on Training Images: ',f1)
print('\nConfusion Matrix: \n',conf_mat)

plt.matshow(conf_mat)
plt.title('Confusion Matrix for Training Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


test_labels_pred = clf.predict(test_img)

acc = accuracy_score(test_labels,test_labels_pred)
tprecision = precision_score(test_labels,test_labels_pred,average='macro')
trecall = recall_score(test_labels,test_labels_pred,average='macro')
tf1 = f1_score(test_labels,test_labels_pred,average='macro')
conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

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
