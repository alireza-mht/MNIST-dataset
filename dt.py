import sys
from mnistloader.mnist_loader import MNIST
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from sklearn.model_selection import cross_val_score
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pydotplus

import io
from sklearn.externals.six import StringIO
from matplotlib import style
style.use('ggplot')

old_stdout = sys.stdout
log_file = open("DTsummary.log","w")
sys.stdout = log_file


print("Decision Tree Log :\n\n")

mndata = MNIST('./mnistloader/dataset/')

trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
# clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=32,min_samples_split=8, min_samples_leaf=8 , max_features=784)

clf = clf.fit(trainingImages[:60000], trainingLabels[:60000])
predictionResTrain = clf.predict(trainingImages)

accuracy = accuracy_score(trainingLabels.tolist(), predictionResTrain)
precision = precision_score(trainingLabels.tolist(), predictionResTrain,average='macro')
recall = recall_score(trainingLabels.tolist(), predictionResTrain,average='macro')
f1 = f1_score(trainingLabels.tolist(), predictionResTrain,average='macro')
conf_mat = confusion_matrix(trainingLabels.tolist(), predictionResTrain)


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


predictionRes = clf.predict(testImages)

acc = accuracy_score(testLabels.tolist(),predictionRes)
tprecision = precision_score(testLabels.tolist(),predictionRes,average='macro')
trecall = recall_score(testLabels.tolist(),predictionRes,average='macro')
tf1 = f1_score(testLabels.tolist(),predictionRes,average='macro')
conf_mat_test = confusion_matrix(testLabels.tolist(),predictionRes)

print('\nAccuracy of Classifier on Test Images: ',acc)
print('\nPrecision of Classifier on Test Images: ',tprecision)
print('\nRecall of Classifier on Test Images: ',trecall)
print('\nF1 of Classifier on Test Images: ',tf1)
print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

sys.stdout = old_stdout
log_file.close()
