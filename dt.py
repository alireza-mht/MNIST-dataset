import sys
from mnistloader.mnist_loader import MNIST
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix

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


print('\nLoading MNIST Data...')

mndata = MNIST('./mnistloader/dataset/')

print('\nLoading Training Data...')

trainingImages, trainingLabels = mndata.load_training()

print('\nLoading Testing Data...')

testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

print('\nPreparing Decision Tree...')

clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
clf = clf.fit(trainingImages[:1000], trainingLabels[:1000])

#clf = clf.fit(trainingImages[:60000], trainingLabels[:60000])


print('\nMaking Predictions on Validation Data...')

predictionRes = clf.predict(testImages)

print( metrics.classification_report(testLabels.tolist(), predictionRes, digits=4))

# Cross Validation Results Exercise 3.3 for Decision Tree
scores = cross_val_score(clf, trainingImages[:1000], trainingLabels[:1000].tolist(), cv=5)
print (scores)
# print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(testLabels.tolist(),predictionRes)
print(accuracy)


print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(testLabels.tolist(),predictionRes)

print('\nConfusion Matrix: \n',conf_mat)
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# # Pixel importances on 28*28 image
# importances = clf.feature_importances_
# importances = importances.reshape((28, 28))

# # Plot pixel importances
# plt.matshow(importances, cmap=plt.cm.hot)
# plt.title("Pixel importances for decision tree")
# plt.show()

sys.stdout = old_stdout
log_file.close()

# Decision Tree as output -> decision_tree.png
dot_data = io.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')
