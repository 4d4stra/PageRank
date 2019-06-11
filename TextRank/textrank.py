import os
import sys
import copy
import collections
import pandas as pd

#import nltk
#import nltk.tokenize
import spacy
nlp=spacy.load('en_core_web_lg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pagerank

'''
    textrank.py
    -----------
    This module implements TextRank, an unsupervised keyword
    significance scoring algorithm. TextRank builds a weighted
    graph representation of a document using words as nodes
    and coocurrence frequencies between pairs of words as edge 
    weights. It then applies PageRank to this graph, and 
    treats the PageRank score of each word as its significance.
    The original research paper proposing this algorithm is
    available here:
    
        https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
'''

## TextRank #####################################################################################
    
def __preprocessDocument_nltk(document, relevantPosTags):#"NN", "ADJ"
    '''
    This function accepts a string representation 
    of a document as input, and returns a tokenized
    list of words corresponding to that document.
    '''
    
    words = __tokenizeWords(document)
    posTags = __tagPartsOfSpeech(words)
    
    # Filter out words with irrelevant POS tags
    filteredWords = []
    for index, word in enumerate(words):
        word = word.lower()
        tag = posTags[index]
        if not __isPunctuation(word) and tag in relevantPosTags:
            filteredWords.append(word)

    return filteredWords

def __preprocessDocument_spacy(document, relevantPosTags,merge_ents=False):
    '''
    This function accepts a string representation 
    of a document as input, and returns a tokenized
    list of words corresponding to that document.
    '''
    doc=nlp(document)
    if merge_ents:
        ents=[i.ent_type_ for i in doc]
    words = [i.text for i in doc]
    posTags = [i.pos_ for i in doc]
    
    # Filter out words with irrelevant POS tags
    filteredWords = []
    typedict={}
    entflag=False
    enttype=None
    entword=None
    for index, word in enumerate(words):
        word = word.lower()
        tag = posTags[index]
        if merge_ents:
            ent = ents[index]
        if merge_ents and not __isPunctuation(word) and tag in relevantPosTags:
            if not entflag and not ent in ['PERSON','ORG']: 
                filteredWords.append(word)
                if not word in typedict.keys():
                    typedict[word]=tag
            elif entflag and ent==enttype:
                if not word in typedict.keys():
                    typedict[word]=tag
                entword='_'.join([entword,word])
            elif not entflag and ent in ['PERSON','ORG']:
                if not word in typedict.keys():
                    typedict[word]=tag
                entword=word
                enttype=ent
                entflag=True
            elif entflag and ent!=enttype:
                filteredWords.append(entword)
                if not entword in typedict.keys():
                    typedict[entword]=enttype
                filteredWords.append(word)
                if not word in typedict.keys():
                    typedict[word]=tag
                enttype=None
                entword=None
                entflag=False
        elif merge_ents and entflag:
            filteredWords.append(entword)
            if not entword in typedict.keys():
                typedict[entword]=enttype
            entflag=False
            enttype=None
            entword=None
        elif not __isPunctuation(word) and tag in relevantPosTags:
            filteredWords.append(word)

    if merge_ents:
        return filteredWords,typedict
    return filteredWords

def create_network(document,windowSize=2, relevantPosTags=["NOUN","ADJ","PROPN"],ngrams=[],merge_ents=False):
    # Tokenize document:
    if not merge_ents:
        words = __preprocessDocument_spacy(document, relevantPosTags,merge_ents=merge_ents)
    else:
        words,typedict = __preprocessDocument_spacy(document, relevantPosTags,merge_ents=merge_ents)
    #words returns a list of adjacent tokens

    #ngram document if using
    for ngram in ngrams:
        words=ngram[words]
    
    # Build a weighted graph where nodes are words and
    # edge weights are the number of times words cooccur
    # within a window of predetermined size. In doing so
    # we double count each coocurrence, but that will not
    # alter relative weights which ultimately determine
    # TextRank scores.
    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    for index, word in enumerate(words):
        for otherIndex in range(index - windowSize, index + windowSize + 1):
            if otherIndex >= 0 and otherIndex < len(words) and otherIndex != index:
                otherWord = words[otherIndex]
                edgeWeights[word][otherWord] += 1.0
                # boosting ngram rank
                if len(ngrams)>0 and "_" in word:
                    w_list=word.split("_")
                    for i in range(len(w_list)-1):
                        k=i+1
                        w1=w_list[i]
                        w2=w_list[k]
                        edgeWeights[w1][w2]+=1.0
                        edgeWeights[word][w1]+=1.0
                        edgeWeights[w2][w1]+=1.0
                        edgeWeights[w1][word]+=1.0
                        if i==len(w_list)-1:
                            edgeWeights[word][w2]+=1.0
                            edgeWeights[w2][word]+=1.0

    #create dataframe
    word_1=[]
    word_2=[]
    type_1=[]
    type_2=[]
    counts=[]
    for key_1 in edgeWeights.keys():
        for key_2 in edgeWeights[key_1].keys():
            word_1.append(key_1)
            word_2.append(key_2)
            counts.append(edgeWeights[key_1][key_2])
            try:
                type_1.append(typedict[key_1])
            except:
                catchflag=False
                for w_i in key_1.split('_')[::]:
                    try:
                        type_1.append(typedict[w_i])
                        catchflag=True
                        break
                    except:
                        continue
                if not catchflag:
                    type_1.append('Unknown')
            try:
                type_2.append(typedict[key_2])
            except:
                catchflag=False
                for w_i in key_2.split('_')[::]:
                    try:
                        type_2.append(typedict[w_i])
                        catchflag=True
                        break
                    except:
                        continue
                if not catchflag:
                    type_2.append('Unknown')
                    
                    
    #print(len(type_1),len(word_1),len(type_2),len(word_2))
    df_counts=pd.DataFrame(data={'w1':word_1,'t1':type_1,'w2':word_2,'t2':type_2,'count':counts})
    df_counts=df_counts.loc[(~df_counts.w1.isnull()) & (~df_counts.t1.isnull()) & (~df_counts.w2.isnull()) & (~df_counts.t2.isnull())]
    return df_counts

def textrank(document, windowSize=2, rsp=0.15, relevantPosTags=["NOUN","ADJ","PROPN"],ngrams=[]):
    '''
    This function accepts a string representation
    of a document and three hyperperameters as input.
    It returns Pandas matrix (that can be treated
    as a dictionary) that maps words in the
    document to their associated TextRank significance
    scores. Note that only words that are classified
    as having relevant POS tags are present in the
    map.
    '''
    
    # Tokenize document:
    words = __preprocessDocument_spacy(document, relevantPosTags)
    #words returns a list of adjacent tokens

    #ngram document if using
    for ngram in ngrams:
        words=ngram[words]
    
    # Build a weighted graph where nodes are words and
    # edge weights are the number of times words cooccur
    # within a window of predetermined size. In doing so
    # we double count each coocurrence, but that will not
    # alter relative weights which ultimately determine
    # TextRank scores.
    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    for index, word in enumerate(words):
        for otherIndex in range(index - windowSize, index + windowSize + 1):
            if otherIndex >= 0 and otherIndex < len(words) and otherIndex != index:
                otherWord = words[otherIndex]
                edgeWeights[word][otherWord] += 1.0
                # boosting ngram rank
                if len(ngrams)>0 and "_" in word:
                    w_list=word.split("_")
                    for i in range(len(w_list)-1):
                        k=i+1
                        w1=w_list[i]
                        w2=w_list[k]
                        edgeWeights[w1][w2]+=1.0
                        edgeWeights[word][w1]+=1.0
                        edgeWeights[w2][w1]+=1.0
                        edgeWeights[w1][word]+=1.0
                        if i==len(w_list)-1:
                            edgeWeights[word][w2]+=1.0
                            edgeWeights[w2][word]+=1.0

    # Apply PageRank to the weighted graph:
    wordProbabilities = pagerank.powerIteration(edgeWeights, rsp=rsp)
    wordProbabilities.sort_values(inplace=True,ascending=False)

    return wordProbabilities


## NLP utilities ################################################################################

def __asciiOnly(string):
    return "".join([char if ord(char) < 128 else "" for char in string])

def __isPunctuation(word):
    return word in [".", "?", "!", ",", "\"", ":", ";", "'", "-","”","“"]

def __tagPartsOfSpeech(words):
    return [pair[1] for pair in nltk.pos_tag(words)]

def __tokenizeWords(sentence):
    return nltk.tokenize.word_tokenize(sentence)

## tests ########################################################################################

def applyTextRank(fileName, title="a document",output=False):
    print("\n")
    print("Reading \"%s\" ..." % title)
    filePath = os.path.join(os.path.dirname(__file__), fileName)
    document = open(filePath).read()
    document = __asciiOnly(document)
    
    print("Applying TextRank to \"%s\" ..." % title)
    keywordScores = textrank(document)

    if output:
        return keywordScores
    
    print("\n")
    header = "Keyword Significance Scores for \"%s\":" % title
    print(header)
    print("-" * len(header))
    print(keywordScores)
    print("\n")
