import pandas as pd
from pathlib import Path
import sklearn.datasets as skds
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
import itertools
import numpy as np
import matplotlib.pyplot as plt

"""Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        NewsArticles/
            Mythology/
                Mythology_file_1.txt
                Mythology_file_2.txt
                ...
                Mythology_file_100.txt
            Politics/
                Politics_file_101.txt
                ...
                Politics_file_200.txt
            Science/
                Science_file_201.txt
                ...
                Science_file_300.txt
            Sports/
                Sports_file_301.txt
                ...
                Sports_file_400.txt
            Technology/
                Technology_file_401.txt
                ...
                Technology_file_500.txt
                ..."""
# Source file directory
path_train = "G:\\DataSet\\DataSet\\NewsArticles"
files_train = skds.load_files(path_train, load_content=False)
label_index = files_train.target
label_names = files_train.target_names
labelled_files = files_train.filenames

data_tags = ["filename", "category", "content"]
data_list = []


# Read and add data from file to a list
i = 0
for f in labelled_files:
    data_list.append((f, label_names[label_index[i]], Path(f).read_text()))
    i += 1

# We have training data available as dictionary filename, category, data
data = pd.DataFrame.from_records(data_list, columns=data_tags)
print("Data loaded : "+str(len(data)))
# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .8)

train_posts = data['content'][:train_size]
train_tags = data['category'][:train_size]
train_files_names = data['filename'][:train_size]

test_posts = data['content'][train_size:]
test_tags = data['category'][train_size:]
test_files_names = data['filename'][train_size:]
print("Training Set Size "+str(len(train_posts)))
print("Testing Set Size "+str(len(test_posts)))
#print(train_tags[:5])

"""
 https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

 https://www.opencodez.com/python/text-classification-using-keras.htm

vectorizer = CountVectorizer(stop_words='english', max_features=500)
"""
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train = vectorizer.fit_transform(train_posts)
vectorizer1 = CountVectorizer(stop_words='english', max_features=500)
X_test = vectorizer.transform(test_posts)
print("Vocab Size : "+str(X_train.shape[0]))
#https://www.opencodez.com/python/text-classification-using-keras.htm
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, train_tags)

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
print("Cross validation start")
print(cross_val_score(clf, X_train, train_tags, cv=k_fold, n_jobs=1))
print("Cross validation end")
print("start applying on testing data set")
predictions = clf.predict(X_test)
print("Model applied on testing data set")
class_names = ["Mythology", "Politics", "Science", "Sports", "Technology"]
print("***********Classification Report***********")
print(classification_report(test_tags, predictions, class_names))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    print("confusion matrix")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_tags, predictions)
# Plot Confusion Matrix

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
# save the model to disk
modelFileName = 'finalized_model.sav'
print("Saving model")
pickle.dump(clf, open(modelFileName, 'wb'))
print("model saved...")

print("load previously saved model")
# load previously saved model from disk
loaded_model = pickle.load(open(modelFileName, 'rb'))
print("Start prediction from loaded model")
predictions = clf.predict(X_test)
print("***********Classification Report***********")
print(classification_report(test_tags, predictions, class_names))
print("complete")
