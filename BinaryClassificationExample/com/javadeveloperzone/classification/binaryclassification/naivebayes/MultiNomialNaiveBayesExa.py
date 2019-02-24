import pandas as pd
from pathlib import Path
import sklearn.datasets as skds
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.model_selection import KFold, cross_val_score

"""Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        EmailSpam/
            Spam/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            ham/
                file_43.txt
                file_44.txt
                ..."""
# Source file directory
path_train = "G:\\DataSet\\DataSet\\EmailSpam\\EmailSpam"
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


# https://www.kaggle.com/adamschroeder/countvectorizer-tfidfvectorizer-predict-comments

# https://www.opencodez.com/python/text-classification-using-keras.htm
#vectorizer = TfidfVectorizer()

vectorizer = CountVectorizer(stop_words='english',max_features=500)
X_train = vectorizer.fit_transform(train_posts)
vectorizer1 = CountVectorizer(stop_words='english',max_features=500)
X_test = vectorizer.fit_transform(test_posts)
print("Vocubalary")
print(X_train[1])
#https://www.opencodez.com/python/text-classification-using-keras.htm
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, train_tags)

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
print("Cross validation start")
print (cross_val_score(clf, X_train, train_tags, cv=k_fold, n_jobs=1))
print("Cross validation end")
print("start applying on testing data set")
predections = clf.predict(X_test)
print("Model applied on testing data set")

print("***********Classification Report***********")
print(classification_report(test_tags,predections,["ham","spam"]))

# save the model to disk
modelfilename = 'finalized_model.sav'
print("Saving model")
pickle.dump(clf, open(modelfilename, 'wb'))
print("model saved...")

print("load previously saved model")
# load previously saved model from disk
loaded_model = pickle.load(open(modelfilename, 'rb'))
print("Start predection from loaded model")
predections = clf.predict(X_test)
print("***********Classification Report***********")
print(classification_report(test_tags,predections,["ham","spam"]))
print("complete")
