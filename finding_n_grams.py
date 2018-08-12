__author__ = 'Nehas'
#date : 08-05-18
#######################################
#            Methods 
#######################################
'''
1. reading from file and storing into container
2. cleaning data
3. fetch to model

'''
import csv
import string
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.summarization import keywords

import nltk
import time
from collections import Counter
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


# ngram Based
#1.1 bigrams from each sentences
#1.2 bigrams from same groups( paragraphs)

sentences = []
bigram_bag = []
bigram = Phrases()

'''1.PRE-PROCESSING (Read from CSV file and clean it)'''
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def collocation():
    with open("sentences.csv", "r") as sentencesfile, open("collocation.csv", 'wb') as f_out:
        reader = csv.reader(sentencesfile, delimiter = ",")
        writer = csv.writer(f_out)
        heading = ["Sentence", "keywords","Bigrams","Trigrams","Four_grams"]
        writer.writerow(heading)
        reader.next()
       
        for row in reader:
            sentence = " ".join(str(x) for x in row)
            token=nltk.word_tokenize(clean(str(row)))
            #token=nltk.word_tokenize(str(sentence))
            sen = Phrases([clean(str(row)).split()]).vocab.keys()
            bigram_data = [key for key in sen if len(key.split("_")) > 1]
            
            trigrams = ngrams(token,3)
            fourgrams = ngrams(token,4)
            trigram_bag = [', '.join(' '.join((a, b, c)) for a, b, c in trigrams)]
            fourgram_bag = [', '.join(' '.join((a, b, c, d)) for a, b, c, d in fourgrams)]
            writer.writerow(row + [keywords(sentence), bigram_data, trigram_bag, fourgram_bag])   

    sentencesfile.close()
    f_out.close()

collocation()
