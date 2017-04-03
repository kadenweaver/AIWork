# Author:Kaden Weaver
# Purpose: compare feature sets for sentiment analysis in movie reviews.
# Citations:



import random
import math
import collections
from nltk.corpus import movie_reviews
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.classify import accuracy

# Divide the corpus for training and testing
train_fileids = movie_reviews.fileids()[0:500] + movie_reviews.fileids()[1000:1500]
test_fileids = movie_reviews.fileids()[500:1000] + movie_reviews.fileids()[1500:2000]

# Define a function for formatting classification datasets
def format_dataset(fileids, featureset):
    dataset = list()
    for fileid in fileids:
        review = set(movie_reviews.words(fileid))
        features = dict()
        for word in featureset:
            features[word] = word in review
        pos_or_neg = fileid[:3]
        dataset.append((features,pos_or_neg))
    return dataset

# Collect all the words in the training examples
vocabulary = set()
for fileid in train_fileids:
    for word in movie_reviews.words(fileid):
        vocabulary.add(word)

# Try a feature set of 500 random words
vocabulary = list(vocabulary)
random.shuffle(vocabulary)
random_featureset = vocabulary[:500]

train_set = format_dataset(train_fileids, random_featureset)
test_set = format_dataset(test_fileids, random_featureset)
bayes = NaiveBayesClassifier.train(train_set)

print("Random words: ", random_featureset)
print("Naive Bayes accuracy:", accuracy(bayes, test_set))


#***************************************************************************
# Try a feature set of the 500 words that appear most often in the training examples
popwords = collections.defaultdict(int)
for fileid in train_fileids:
    review = set(movie_reviews.words(fileid))
    for word in review:
        popwords[word] = popwords[word] + 1
popvocab = list()
for n in range(500):
    maxword = max(popwords, key=popwords.get)
    popvocab.append(maxword)
    del popwords[maxword]
poptrain_set = format_dataset(train_fileids, popvocab)
poptest_set = format_dataset(test_fileids, popvocab)
pop = NaiveBayesClassifier.train(poptrain_set)
print("Popular words: ", popvocab)
print("Popwords accuracy: " ,accuracy(pop,poptest_set))




#********************************************************
# Try a feature set of the 500 words with highest information gain on the training examples


posrev = 0
negrev = 0


infodict = collections.defaultdict(lambda: (0,0))

for fileid in train_fileids:
    review = set(movie_reviews.words(fileid))
    for word in review:
        post,negt = infodict[word]
        if fileid[:3] == 'pos':
            infodict[word] = (post+1,negt)
            posrev += 1
        else:
            infodict[word] = (post, negt+1)
            negrev += 1

    


# Compute the entropy of a dataset with p positive
# examples and n negative examples.
def entropy(p, n):
    fp = 0 if p==0 else p / (p+n)
    fn = 0 if n==0 else n / (p+n)
    plog = 0 if fp==0 else fp*math.log(fp,2)
    nlog = 0 if fn==0 else fn*math.log(fn,2)
    return -plog -nlog

# Compute the information gain if we split a dataset
# with p positives and n negatives into two datasets
# (one with p_true and n_true, one with p_false and n_false)
def gain(p, n, p_true, n_true, p_false, n_false):
    f_true = (p_true + n_true) / (p+n)
    f_false = (p_false + n_false) / (p+n)
    e = entropy(p, n)
    e_true = entropy(p_true, n_true)
    e_false = entropy(p_false, n_false)
    return e - f_true*e_true - f_false*e_false


#*******************************************************
infogaindict = collections.defaultdict(float)

for word in infodict:
    ptrue, ntrue = infodict[word]
    pfalse = posrev - ptrue
    nfalse = negrev - ntrue
    infogain = gain(posrev,negrev,ptrue,ntrue,pfalse,nfalse)
    infogaindict[word] = infogain

infovocab = list()
for n in range(500):
    maxword = max(infogaindict, key=infogaindict.get)
    infovocab.append(maxword)
    del infogaindict[maxword]

infotrain_set = format_dataset(train_fileids, infovocab)
infotest_set = format_dataset(test_fileids, infovocab)
info = NaiveBayesClassifier.train(infotrain_set)
print("Info words: ", infovocab)
print("Infowords accuracy: " ,accuracy(info,infotest_set))


