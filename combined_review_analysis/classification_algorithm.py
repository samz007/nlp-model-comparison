from pip._vendor.distlib.compat import raw_input


def get_comment_type_positive_or_negative():
    comment = raw_input("Enter the review comment given by customer  : ")
    with open("imdb_labelled.txt", "r") as text_file:
        lines = text_file.read().split('\n')
    with open("yelp_labelled.txt", "r") as text_file:
        lines += text_file.read().split('\n')
    with open("amazon_cells_labelled.txt", "r") as text_file:
        lines += text_file.read().split('\n')
    lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")[1] != '']
    train_documents = [line[0] for line in lines]
    train_labels = [int(line[1]) for line in lines]
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer(binary='true')
    train_documents = count_vectorizer.fit_transform(train_documents)
    from sklearn.naive_bayes import BernoulliNB
    classifier = BernoulliNB().fit(train_documents, train_labels)
    if(classifier.predict(count_vectorizer.transform([comment]))==1):
        print("customer is satisfied")
    else:
        print("Result: customer is not satisfied")

get_comment_type_positive_or_negative()
