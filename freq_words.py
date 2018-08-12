__author__ = 'Nehas'
# coding: utf-8

'''
pre_req:
        spacy --> 2.0.11
'''

######################
# phrases extraction
######################
import re
import spacy
import numpy as np
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()

s_words = ["from", "can", "we", "are", "an", "it", "of", "and", "their",
             "me", "those", "on", "for", "be", "is", "with", "was", "what",
             "okay", "i'm", "my", "thanks", "or", "am", "i'll", "hello", "hi",
             "the", "in", "to", "these", "ok", "that", "at", "this", "i", "how", "bye", "a"]

spacy_stop_words = ["-pron-", "-", "'s", "-PRON-"]

special_chars = re.compile("[@,\.{}()\[\]!?<>0-9/-:=;_\|]")

def clean_sentence_no_lemma(sentence):
        sentence = sentence.lower()
        sentence = re.sub('\s+',  ' ', sentence) #matches any whitespace characte
        sentence = re.sub('\s+$|^\s+', '', sentence) #remove whitespace from start of the line and end of the line
        sentence = re.sub(r'[^\x00-\x7f]', r'', sentence) #a single character in the range between  (index 0) and  (index 127) (case sensitive)
        sentence = sentence.strip(""" '!:?-_().,'"[]{};*""")
        sentence = ' '.join([w.strip(""" '!:?-_().,'"[]{};*""") for w in re.split(' ', sentence)])
        sentence = sentence.lower()
        sentence = re.sub('  ', ' ', sentence)
        return sentence

def lemmatize(t):
    #return t.lemma_ if t.lemma_ != '-PRON-' else t.text
    return t.lemma_

def get_verb_objects(sentence):
    sentence = clean_sentence_no_lemma(sentence)
    doc = nlp(sentence)

    tokens = []
    intents = []
    for token in doc:
        tokens.append((lemmatize(token), token.tag_, token.dep_, lemmatize(token.head), token.head.tag_))

    for t in tokens:
        if (t[2] == "dobj"): # or t[2] == "compound"):
            st = t[3] + ' ' + t[0]

            vo_words = [token for token in re.split(' ', st) if token.lower() not in spacy_stop_words and token.lower() not in s_words and not special_chars.search(token.lower())]
            intents.append(' '.join(vo_words))

    return intents

def extract_noun_phrases(sentence):
    sentence= clean_sentence_no_lemma(sentence)
    doc = nlp(sentence)

    nps = []
    for np in doc.noun_chunks:
        words = re.split(' ', np.lemma_)

        np_words = [token for token in re.split(' ', np.lemma_) if token.lower() not in spacy_stop_words and token.lower() not in s_words and not special_chars.search(token.lower())]

        if (len(np_words) >= 1):
            np_w = ' '.join(np_words)
            nps.append(np_w)

    return nps

# Testing 
sentence = "How do I pay a supplier via Bpay"
print (get_verb_objects(sentence))
print (extract_noun_phrases(sentence))

import pandas as pd
df = pd.read_csv("model_predicted_output.csv")

print('\ntotal entries =' +str(df.shape))
df = df.drop(df[df.Tag== 'log_snippet'].index)
df = df.drop(df[df.Tag == 'diagnostic_info'].index)
print(df)

summary = df['summary']
np = []
vo = []
phrases = []
for did,content in summary.iteritems():
    vo.append(get_verb_objects(content))
    np.append(extract_noun_phrases(content))
    if get_verb_objects(content):
        phrases.append(''.join(get_verb_objects(content)))
    if extract_noun_phrases(content):
        phrases.append(''.join(extract_noun_phrases(content)))
    
print(phrases)  

df2 = pd.DataFrame(data={'content':sentence, 'noun_phase': np, 'verb_object': vo})
print(df2.head(5))
df2.to_csv("phrases_collection.csv", index=False , encoding='utf-8')


# # Frequency Profile of VO Doubles
from collections import Counter
intents_only = [intent for intent in phrases]
fp = Counter(intents_only)
fp.most_common(50)
print(fp)

for k, v in fp.most_common(10):
    print((k, v))
len(intents_only)


# # KMeans Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import math

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1, 2))
X = vectorizer.fit_transform(intents_only)
true_k = 15
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

##################
#working with IDF
##################
feature_names = vectorizer.get_feature_names()

#idf
print(X)
print("Top terms per cluster:")
print()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print("---" * 10)
    print()

##########
# # LSA
##########
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
X = vectorizer.fit_transform(intents_only)
lsa = TruncatedSVD(n_components=20, n_iter=100)
lsa.fit(X)
terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print ('Concept {}: \n'.format(i))
    for term in sortedTerms:
        print (term[0])
    print("---"*20)
    print()

measurements = [{'city': 'Dubai', 'temperature': 33.},{'city': 'London', 'temperature': 12.},{'city': 'San Francisco', 'temperature': 18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())