import pandas as pd
from gensim.summarization import summarize
from gensim.summarization import keywords

summarize_list = []
keywords_list = []

# input.xlsl is excel file with 2 column consisting for paragraphs and its tag
# output list of summaried sentenses and keywords from the pargraphs 

df = pd.read_excel('input.xlsx')
pd.options.display.max_colwidth = 300
grouped = df.groupby('Tag')     #Tag is the name of column
for name,group in grouped:
    cluster = u''.join(group['category_1'].to_string(index=False)).encode('utf-8').strip()
    if len(cluster) > 50:       # less sentense in a paragraph does not generate summary 
        if name == 'abc' or name == 'xyz' or name == 'pqr':    #Tags in category_1 group  
            a = summarize(cluster, word_count=20)
            summarize_list.append(a)
            keywords_list.append(keywords(cluster))
    else:
        pass
    print "\n"

print summarize_list
print keywords_list