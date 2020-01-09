# NLP-model-comparison

**About Natural Language Processing (NLP)**

One of the most significant features that AI comes with is NLP. 
NLP helps developers to organize and structure knowledge to perform tasks like translation, summarization, named entity recognition, relationship extraction, speech recognition, topic segmentation, etc.
Words used in language and their arrangement in sentences are related to the feelings involved. The mathematics behind these words in sentences can be used to derive predictions about the sentiments of the language used.

**Data**

Dataset for model comparison can be obtained at [Kaggle platform](https://www.kaggle.com/bittlingmayer/amazonreviews). 
The dataset is available in the form of sentences with corresponding classifier labels. Train data is used to train the classifier while
Test data can be used to evaluate the performances of different NLP classifier models.

**Project**

This project covers the `Semantic Analysis` aspect of NLP. Semantic Analysis is a structure created by the syntactic analyzer which assigns meanings. In Semantic Analysis, linear sequences of words are transferred into structures. It shows how the terms are associated with each other.
The project consists of the following components:

* [Algorithm designing and Model comparison using Amazon review comments](https://github.com/samz007/nlp-model-comparison/tree/master/review_comment_analysis.ipynb): In this part, a relatively large dataset of review comments is preprocessed, analyzed, and used to train different classifiers. The models were tested against different test sets of review comments, and their performances were compared.

* [Review analysis for restaurant review comments](https://github.com/samz007/nlp-model-comparison/tree/master/restaurant_review_analysis): In this part, review comments regarding the quality of food given at restaurants were used to train different classifier models, and their performances were analyzed.

* [Sentiment analysis on combined data](https://github.com/samz007/nlp-model-comparison/tree/master/combined_review_analysis): In this part, review comments obtained for movies and online-shopping platform were combined to train the classifier. The effect of the combination was analyzed, and the prediction model was built to predict the satisfaction level of a customer after inputting the comment.


**References**

* NLTK library overview: https://www.nltk.org/book/ch06.html
* NLTK classifiers: https://www.nltk.org/api/nltk.classify.html
* NLTK official git repo: https://github.com/nltk/nltk
* Transforming a count matrix to a normalized tf or tf-idf representation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html