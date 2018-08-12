__author__ = 'Nehas'
# coding: utf-8

######################
# phrases extraction
######################
'''
pre_req:
        spacy --> 2.0.11
'''

import re,json
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

phrases_dict = {}
query = input("Enter the query: ")
np = extract_noun_phrases(query)
vo = get_verb_objects(query)
phrases_dict[query] = np,vo
json = json.dumps(phrases_dict)
print(json)

