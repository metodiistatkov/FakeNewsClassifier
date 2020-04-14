import pandas
import re
from langdetect import detect
import nltk
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn import metrics, tree
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

# download nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')

# disable default panda warnings that are printed in the console output
pandas.options.mode.chained_assignment = None

train_set_tweets = pandas.read_csv('mediaeval-2015-trainingset.txt', sep='\\t', engine='python')
test_set_tweets = pandas.read_csv('mediaeval-2015-testset.txt', sep='\\t', engine='python')

tweets = train_set_tweets.copy(deep=True)


def remove_non_english_tweets(tweets) -> None:
    for i in range(0, len(tweets)):
        try:
            if detect(tweets.tweetText[i]) != 'en':
                tweets.drop(i, inplace=True)
        except:
            continue


def reset_index(tweets) -> None:
    tweets.reset_index(drop=True, inplace=True)


def remove_http(tweets):
    for i in range(0, len(tweets.tweetText)):
        tweets.tweetText[i] = re.sub(r'https?:.*', '', tweets.tweetText[i])


def remove_retweets(tweets):
    for i in range(0, len(tweets.tweetText)):
        if 'RT' in tweets.tweetText[i]:
            tweets.drop(i, inplace=True)


def make_humor_fake(tweets):
    for i in range(0, len(tweets)):
        if tweets.label[i] == 'humor':
            tweets.label[i] = 'fake'


def stem_words(tweets):
    stemmer = nltk.stem.porter.PorterStemmer()
    for i in range(0, len(tweets.tweetText)):
        tokenize = word_tokenize(tweets.tweetText[i])
        for i in range(0, len(tokenize)):
            unstemmed = tokenize[i]
            tokenize[i] = stemmer.stem(unstemmed)
        tweets.tweetText[i] = TreebankWordDetokenizer().detokenize(tokenize)


def preprocess(tweets):
    make_humor_fake(tweets)
    # remove_non_english_tweets(tweets)
    # reset_index(tweets)
    remove_http(tweets)
    remove_retweets(tweets)
    reset_index(tweets)
    stem_words(tweets)


def grid_search_params(algorithm, parameters, x_train, y_train):
    clf = GridSearchCV(algorithm, parameters)
    clf.fit(x_train, y_train)
    print(clf.best_params_)


def train_clf(tweets, algorithm):
    x_train = tweets.tweetText.values
    y_train = tweets.label.values
    x_test = test_set_tweets.tweetText.values
    y_test = test_set_tweets.label.values

    # Alternative to using a pipeline
    #######################################################
    # vectorizer = TfidfVectorizer(stop_words='english')  #
    # train_vectors = vectorizer.fit_transform(x_train)   #
    # test_vectors = vectorizer.transform(x_test)         #
    # print(vectorizer.get_feature_names())               #
    # clf = SVC(gamma='scale').fit(train_vectors, y_train)#
    # predicted = clf.predict(test_vectors)               #
    #######################################################

    tweets_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 1), analyzer='word')),
        ('clf', algorithm),
    ])
    tweets_clf.fit(x_train, y_train)
    predicted = tweets_clf.predict(x_test)
    print(classification_report(y_test, predicted, target_names=["fake", "real"]))


preprocess(tweets)

# Grid Search Example
# parameters = {'fit_prior': ('True', 'False'), 'alpha': [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
# grid_search_params(MultinomialNB(), parameters, vectorizer.fit_transform(tweets.tweetText.values), tweets.label.values)

# Naive Bayes classifier
train_clf(tweets, MultinomialNB(alpha=1.0, fit_prior=True))

# SGDClassifier
# train_clf(tweets, SGDClassifier())

# SVM classifier (SVC)
# train_clf(tweets, SVC(gamma='scale', C=12, coef0=1, kernel='rbf'))

# Linear SVM classifier (LinearSVC)
# train_clf(tweets, LinearSVC())

# RandomForestClassifier
# train_clf(tweets, RandomForestClassifier(max_depth=4, random_state=2))
