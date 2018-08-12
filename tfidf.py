
# coding: utf-8

#feature_extraction - CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer                     

corpus = ['This is the first document.','This is the second second document.', 'And the third one.','Is this the first document?']
X = vectorizer.fit_transform(corpus)
print(X)

analyze = vectorizer.build_analyzer()
analyze("This is a text document to analyze.") == (['this', 'is', 'text', 'document', 'to', 'analyze'])
vectorizer.get_feature_names() == (['and', 'document', 'first', 'is', 'one','second', 'the', 'third', 'this'])
X.toarray() 

#converse mapping from feature name to column index is stored in the vocabulary_ attribute of the vectorizer (pos)
vectorizer.vocabulary_.get('first')

#words that were not seen in the training corpus will be completely ignored in future calls to the transform method
vectorizer.transform(['Something completely new document ']).toarray()

# working with bigrams 
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
analyze('Bi-grams are cool!') == ( ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])

X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(X_2)
print("\n Get the vocab position ")
print(bigram_vectorizer.vocabulary_)

# TF-IDF
'''
In order to re-weight the count features into floating point 
values suitable for usage by a classifier it is very common to use the tf–idf transform.
TfidfTransformer’s default settings, TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
'''
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

counts = [[3, 0, 1],[2, 0, 0],[3, 0, 0],[4, 0, 0],[3, 2, 0],[3, 0, 2]]
tfidf = transformer.fit_transform(counts)
print(tfidf)
print()
print(tfidf.toarray())
#output (doc, term)
#tfidf.toarray()

transformer = TfidfTransformer()
transformer.fit_transform(counts).toarray()

'''
So although both the CountVectorizer and TfidfTransformer (with use_idf=False) 
produce term frequencies, TfidfTransformer is normalizing the count.
'''

#TfidfVectorizer that combines all the options of CountVectorizer and TfidfTransformer in a single model:

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(corpus)
tfidf.toarray() 