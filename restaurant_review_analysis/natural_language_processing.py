# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning reviews
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import    Pipeline

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X =cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



#creating different models for classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
classifier_RF  = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_NB = GaussianNB()
classifier_SVC = SVC(kernel='rbf',random_state=0, gamma='auto')
classifiers = [ classifier_RF, classifier_DT, classifier_NB , classifier_SVC ]
classifier_names  = [ 'random forest','decision tree', 'naive bayes', 'support vector machine rbf kernel']
i = 0
for  classifier in classifiers:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test, y_pred, average='micro')
    print( classifier_names[i]+ ": " +np.str(np.round(score,  decimals=3)))
    i+=1
