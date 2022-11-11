import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier

train = pd.read_csv('data/train.txt', header=None, names=['sentence', 'emotion'], sep=';')
val = pd.read_csv('data/val.txt', header=None, names=['sentence', 'emotion'], sep=';')
test = pd.read_csv('data/test.txt', header=None, names=['sentence', 'emotion'], sep=';')

print('**Preprocessing Step Has Started**')
train_input = list(train['sentence'])
train_target = list(train['emotion'])

val_input = list(val['sentence'])
val_target = list(val['emotion'])

test_input = list(test['sentence'])
test_target = list(test['emotion'])

def make_token_bags(input_data):
    token_bags = []
    for sentence in input_data:
        token_bag = []
        tokens = word_tokenize(sentence)
        for token in tokens:
            if token not in stopwords.words('english'):
                token_bag.append(token)
        token_bags.append(token_bag)
    return token_bags

train_token_bags = make_token_bags(train_input)
val_token_bags = make_token_bags(val_input)
test_token_bags = make_token_bags(test_input)

entire_bag = list(itertools.chain(*train_token_bags))
unique_words = np.unique(entire_bag)

ids = range(len(unique_words))

w2id = dict(zip(unique_words, ids))

train_mat = np.zeros(shape=(len(train_input), len(unique_words)))
val_mat = np.zeros(shape=(len(val_input), len(unique_words)))
test_mat = np.zeros(shape=(len(test_input), len(unique_words)))

def update_matrix(matrix, token_bags, mapping):
    for i, tokens in enumerate(token_bags):
        for token in tokens:
            if token in mapping.keys():
                matrix[i][mapping[token]] += 1
            else:
                pass
    return matrix

train_mat = update_matrix(train_mat, train_token_bags, w2id)
val_mat = update_matrix(val_mat, val_token_bags, w2id)
test_mat = update_matrix(test_mat, test_token_bags, w2id)

def get_tf(matrix, unique_words):
    tf = matrix/len(unique_words)
    return tf

train_tf = get_tf(train_mat, unique_words)
val_tf = get_tf(val_mat, unique_words)
test_tf = get_tf(test_mat, unique_words)

def get_idf(unique_words, input_data):
    count = {}
    for word in unique_words:
        count[word] = 0
        for sentence in input_data:
            if word in sentence:
                count[word] += 1
    idf = []
    for i in count.values():
        if i != 0:
            idf.append(np.log(len(input_data)/i))
        else:
            idf.append(np.log(len(input_data)/1e-10))
    idf = np.array(idf)
    return idf

train_idf = get_idf(unique_words, train_input)
val_idf = get_idf(unique_words, val_input)
test_idf = get_idf(unique_words, test_input)

train_tfidf = train_tf * train_idf
val_tfidf = val_tf * val_idf
test_tfidf = test_tf * test_idf
print('**Preprocessing Step Has Completed**')

# Gaussian Naive Bayes
param_smoothing = np.logspace(3,-5,num=9)
train_scores = []
val_scores = []

for i in param_smoothing:
    gnb = GaussianNB(var_smoothing=i).fit(train_tfidf, train_target)
    train_scores.append(gnb.score(train_tfidf, train_target))
    val_scores.append(gnb.score(val_tfidf, val_target))

plt.title('Accuracy Plot by Different Values of var_smoothing')
plt.plot(param_smoothing, train_scores, color='blue')
plt.plot(param_smoothing, val_scores, color='red')
plt.legend(['Train Scores', 'Validation Scores'])
plt.show()

print('**We will Test Naive Bayes Model**')
gnb = GaussianNB(var_smoothing=1).fit(train_tfidf, train_target)
print('--Train Score of Gaussian Naive Bayes Classifier: ', gnb.score(train_tfidf, train_target))
print('--Validation Score of Gaussian Naive Bayes Classifier: ', gnb.score(val_tfidf, val_target))
print('--Test Score of Gaussian Naive Bayes Classifier: ', gnb.score(test_tfidf, test_target))

gnb_test_pred = gnb.predict(test_tfidf)
print('\n-Confusion Matrix: \n', confusion_matrix(test_target, gnb_test_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(test_target, gnb_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

gnb = GaussianNB(var_smoothing=1)
gnb_ovr = OneVsRestClassifier(gnb)
gnb_ovr.fit(train_tfidf, train_target)
gnb_ovr_test_pred = gnb_ovr.predict(test_tfidf)
print('--Test Score of Gaussian Naive Bayes Classifier (OVR): ', gnb_ovr.score(test_tfidf, test_target))
print('\n-Confusion Matrix (OVR): \n', confusion_matrix(test_target, gnb_ovr_test_pred))
print('\n-Precision, Recall, F1-score (OVR): \n', classification_report(test_target, gnb_ovr_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))


# Logistic Regression
param_c = np.logspace(4,-5,num=10)
train_scores = []
val_scores = []

for i in param_c:
    lr = LogisticRegression(multi_class='multinomial', C=i, max_iter=500).fit(train_tfidf, train_target) # Using Softmax function for multinomial classification
    train_scores.append(lr.score(train_tfidf, train_target))
    val_scores.append(lr.score(val_tfidf, val_target))

plt.title('Accuracy Plot by Different Values of C')
plt.plot(param_c, train_scores, color='blue')
plt.plot(param_c, val_scores, color='red')
plt.legend(['Train Scores', 'Validation Scores'])
plt.show()

print('**We will Test Logistic Regression Model**')
lr = LogisticRegression(multi_class='multinomial', C=36000).fit(train_tfidf, train_target)
print('--Train Score of Logistic Regression Classifier: ', lr.score(train_tfidf, train_target))
print('--Validation Score of Logistic Regression Classifier: ', lr.score(val_tfidf, val_target))
print('--Test Score of Logistic Regression Classifier: ', lr.score(test_tfidf, test_target))

lr_test_pred = lr.predict(test_tfidf)
print('\n-Confusion Matrix: \n', confusion_matrix(test_target, lr_test_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(test_target, lr_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

lr_ovr = LogisticRegression(multi_class='ovr', C=36000).fit(train_tfidf, train_target) # One vs. Rest
lr_ovr_test_pred = lr_ovr.predict(test_tfidf)
print('--Test Score of Gaussian Naive Bayes Classifier (OVR): ', lr_ovr.score(test_tfidf, test_target))
print('\n-Confusion Matrix (OVR): \n', confusion_matrix(test_target, lr_ovr_test_pred))
print('\n-Precision, Recall, F1-score (OVR): \n', classification_report(test_target, lr_ovr_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

# Random Forest Classifier
num_trees = [20, 30, 40, 50, 100, 200]
train_scores =[]
val_scores = []

for i in num_trees:
    rf = RandomForestClassifier(min_samples_leaf=30, n_estimators=i).fit(train_tfidf, train_target)
    train_scores.append(rf.score(train_tfidf, train_target))
    val_scores.append(rf.score(val_tfidf, val_target))

plt.title('Accuracy Plot by Different Values of n_estimators')
plt.plot(num_trees, train_scores, color='blue')
plt.plot(num_trees, val_scores, color='red')
plt.legend(['Train Scores', 'Validation Scores'])
plt.show()

print('**We will Test Random Forest Model**')
rf = RandomForestClassifier(min_samples_leaf=3, n_estimators=50).fit(train_tfidf, train_target)
print('--Train Score of Random Forest Classifier: ', rf.score(train_tfidf, train_target))
print('--Validation Score of Random Forest Classifier: ', rf.score(val_tfidf, val_target))
print('--Test Score of Random Forest Classifier: ', rf.score(test_tfidf, test_target))

rf_test_pred = rf.predict(test_tfidf)
print('\n-Confusion Matrix: \n', confusion_matrix(test_target, rf_test_pred))
print('\n-Precision, Recall, F1-score: \n', classification_report(test_target, rf_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

rf = RandomForestClassifier(min_samples_leaf=3, n_estimators=50)
rf_ovr = OneVsRestClassifier(rf)
rf_ovr.fit(train_tfidf, train_target)
rf_ovr_test_pred = rf_ovr.predict(test_tfidf)
print('--Test Score of Random Forest Classifier (OVR): ', rf_ovr.score(test_tfidf, test_target))
print('\n-Confusion Matrix (OVR): \n', confusion_matrix(test_target, rf_ovr_test_pred))
print('\n-Precision, Recall, F1-score (OVR): \n', classification_report(test_target, rf_ovr_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

# SVM
pca = PCA(n_components=100)
pca.fit(train_tfidf)
train_pca = pca.transform(train_tfidf)
val_pca = pca.transform(val_tfidf)
test_pca = pca.transform(test_tfidf)

param_gamma = np.logspace(-1, 9, num=11)
train_scores = []
val_scores = []

for i in param_gamma:
    svm = SVC(kernel='rbf', gamma=i)
    svm_ovr = OneVsRestClassifier(svm)
    svm_ovr.fit(train_pca, train_target)
    train_scores.append(svm_ovr.score(train_pca, train_target))
    val_scores.append(svm_ovr.score(val_pca, val_target))

plt.title('Accuracy Plot by Different Values of gamma')
plt.plot(param_gamma, train_scores, color='blue')
plt.plot(param_gamma, val_scores, color='red')
plt.legend(['Train Scores', 'Validation Scores'])
plt.show()

print('**We will Test Support Vector Machine Model**')
svm = SVC(kernel='rbf', gamma=1e5)
svm_ovr = OneVsRestClassifier(svm)
svm_ovr.fit(train_pca, train_target)

svm_ovr_test_pred = svm_ovr.predict(test_pca)

print('--Test Score of Support Vector Machine (OVR): ', svm_ovr.score(test_pca, test_target))
print('\n-Confusion Matrix (OVR): \n', confusion_matrix(test_target, svm_ovr_test_pred))
print('\n-Precision, Recall, F1-score (OVR): \n', classification_report(test_target, svm_ovr_test_pred, labels=['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']))

