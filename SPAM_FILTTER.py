import csv
import random
from nltk import word_tokenize, WordNetLemmatizer
from nltk import NaiveBayesClassifier, classify
from nltk.corpus import stopwords

#get_stoplist
stoplist=stopwords.words("english")

#preprossing_the_data
def preprocess(email):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(email,errors='ignore'))]

#feature_extraction
def get_features(text):
    return {word: True for word in preprocess(text) if not word in stoplist}

#tain_classifer
def train(all_features,ratio):
    train_size=int(len(all_features)*ratio)
    train_set, test_set = all_features[:train_size], all_features[train_size:]
    clf=NaiveBayesClassifier.train(train_set)
    return train_set, test_set, clf

#evaluate
def evaluate(train_set, test_set, clf):
    print ('Accuracy on the training set = ' + str(classify.accuracy(clf, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(clf, test_set)))
    clf.show_most_informative_features(20)

#Loading_the_data
with open('/Users/badalsatyarthi/Downloads/spam.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    y=[]
    X=[]
    for row in readCSV:
        y.append(row[0])
        X.append(row[1])
        
    y.remove(y[0])
    X.remove(X[0])

all_emails=[(X[i],y[i]) for i in range(len(X))] 
random.shuffle(all_emails)


#preprocess
all_features = [(get_features(email), label) for (email, label) in all_emails]

#training
train_set, test_set, clf = train(all_features, 0.8)

#evalute
evaluate(train_set, test_set, clf)


