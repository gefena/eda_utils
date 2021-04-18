
import numpy as np
from numpy import median
import pandas as pd


import itertools

register_matplotlib_converters()

from nltk.tokenize import word_tokenize
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words

sp = spacy.load('en_core_web_sm')
#sp = spacy.load('en_core_web_lg')
#spacy_stopwords = sp.Defaults.stop_words

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud

import time
import glob, os
from IPython.display import display, Markdown

from datasketch import MinHash, MinHashLSH
from nltk import ngrams
import editdistance
import re

import multiprocessing
from multiprocessing import Pool




def explore_num_words_for_col(df, col):
    num_words_col = col+"_num_words"
    df[num_words_col] = df[col].str.split().str.len()
    #print(num_words_col + ":")
    fig, ax = plt.subplots(figsize=(14,6))
    df[num_words_col].hist(bins=50)
    plt.title(num_words_col)
    plt.show()   
    
    

def generate_wordcloud_for_col(df, col):
    text_data = df[col].astype('str')
    text_for_wordcloud = ','.join(text_data).lower()
    
    #stopwords_set = set(stopwords.split(","))
    wordcloud = WordCloud(width = 1000, height = 400, 
                    background_color ='white', 
                    stopwords = spacy_stop_words, 
                    min_font_size = 12).generate(text_for_wordcloud) 

    # Generate wordcloud in an image file
    #WordCloud.to_file(wordcloud, "wordcloud.png")

    # plot the WordCloud image                        
    plt.figure(figsize = (20, 10), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()  
    
    
def get_top_word_frequency_col(df, col):
    text_data = df[col].fillna("")
    co = CountVectorizer(stop_words=spacy_stop_words)
    counts = co.fit_transform(text_data)
    print("Top Word frequencies:")
    temp_df = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(20)
    temp_df.columns = ["count"]
    display(temp_df) 
    
    
def get_top_bigrams_frequency_col(df, col):
    text_data = df[col].fillna("")
    co = CountVectorizer(ngram_range=(2,2),stop_words=spacy_stop_words)
    counts = co.fit_transform(text_data)
    print("Top Bigrams frequencies:")
    temp_df = pd.DataFrame(counts.sum(axis=0),columns=co.get_feature_names()).T.sort_values(0,ascending=False).head(20)
    temp_df.columns = ["count"]
    display(temp_df)     

    
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([(feature_names[i])
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
        
def simple_topic_modeling_for_col(df, col, num_topic, top_words_for_topic):
    text_data = df[col].fillna("")
    vectorizer = CountVectorizer(stop_words=spacy_stop_words)
    model = vectorizer.fit(text_data)
    docs = vectorizer.transform(text_data)
    lda = LatentDirichletAllocation(num_topic)
    lda.fit(docs)
    
    print_top_words(lda,vectorizer.get_feature_names(),top_words_for_topic)
     
    topic_col = col + '_topic'   
    df[topic_col]=lda.transform(docs).argmax(axis=1)
    series_tmp = df[topic_col].value_counts()
    #data.topic.value_counts(normalize=True).plot.bar()      
    series_tmp.plot(kind="bar", title= "Counts of " + topic_col, figsize=(20,6))
    plt.show()
    
    
    
##############################################################
# Clean text
##############################################################    
        

def clean_text(text, extra_stop_words=''):
    
    stopchar = "~!@#$%^&*(){}[]|,."

    stopwords = extra_stop_words
    stopwords = "," + stopwords + "," + ','.join(spacy_stop_words) + ","

    text2 = text.strip().lower()
    text3 = ''.join([s if not (s in stopchar) else " " for s in text2])
    text4 = ''.join([i for i in text3 if not i.isdigit()])
    text5 = ' '.join(text4.split())
    text6 = ' '.join([w for w in text5.split() if len(w)>2])
    text7 = ' '.join([w for w in text6.split() if ','+w+',' not in stopwords])
    #if len(text6) > 2:
    #    return(text6)
    #else:
    #    return(text4)    
    return(text7)

def clean_text_for_columns(df, col, extra_stop_words=''):
    data = df[col].astype('str')
    joined_text = '[,]'.join(data).lower()
    cleaned_text = clean_text(text, extra_stop_words)
    return(cleaned_text)

def clean_text_column_return_new_col(df, col, new_col):
    df[new_col] = df[col].apply(lambda x: clean_text(str(x)))
    return(df)

##############################################################
# unique text clusters
##############################################################  

def get_minhashes_of_unique_str_list(unique_str_list):

    t0 = time.time()
    # Create an MinHashLSH index optimized for Jaccard threshold 0.5,
    # that accepts MinHash objects with 128 permutations functions
    threshold=0.7
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Create MinHash objects
    minhashes = {}
    for i, s in enumerate(unique_str_list):
        minhash = MinHash(num_perm=128)
        for d in ngrams(s, 3):
            minhash.update("".join(d).encode('utf-8'))
        lsh.insert(i, minhash)
        minhashes[i] = minhash

        if i%5000==0:
            print("counter:",i)
            elapsed_time = time.time() - t0
            print("[exp msg] elapsed time for subprocess: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 


    elapsed_time = time.time() - t0
    print("[exp msg] elapsed time for process: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 

    #for i in range(len(minhashes.keys())):
    #for i in range(10):    
    #    result = lsh.query(minhashes[i])
    #    print("Candidates with Jaccard similarity > " + str(threshold) + " for input", i, ":", result)
    return(lsh, minhashes)

def get_pair_comparisons_list(lsh, minhashes):
    pair_comparisons_list = [] #set()
    singelton_list = []
    for i in range(len(minhashes.keys())):   
        result = lsh.query(minhashes[i])
        if len(result) == 1:
            singelton_list.append(i)
        for j in result:
            if i < j:
                pair_comparisons_list.append((i,j))

        if i%100000==0:
            print("counter:",i)        

    print("len(singelton_list):",len(singelton_list))        
    print("len(pair_comparisons_list):",len(pair_comparisons_list))
    #print(singelton_list[0:5])
    #print(pair_comparisons_list[0:5])
    return(singelton_list, pair_comparisons_list)

def get_text_cluster_network(singelton_list, pair_comparisons_list, unique_str_list):
    set_connected_components = set()
    G=nx.Graph()
    #for first, second in itertools.combinations(descriptions_to_explore, 2):
    for pair in pair_comparisons_list:
        #print(pair)
        first = unique_str_list[pair[0]]
        second = unique_str_list[pair[1]]

        ratio_len = len(first)/len(second)
        max_len = max(len(first), len(second))
        if ratio_len < 1.3 or ratio_len > 0.7:
            if editdistance.eval(first, second) / max_len < 0.1:
                G.add_edge(first, second)
                set_connected_components.add(first)
                set_connected_components.add(second)
                #print("first:", first)
                #print("second:", second)
                ##merged.append(first)

    print("set_connected_components:", len(set_connected_components))
    #set_connected_components_diff = set(descriptions_to_explore).difference(set_connected_components)

    #for n in set_connected_components_diff:
    for i in singelton_list:
        n = unique_str_list[i]   
        G.add_node(n)
    #print("set_connected_components_diff:", len(set_connected_components_diff))

    n = G.number_of_nodes()
    m = G.number_of_edges() 
    ncc = nx.number_connected_components(G)
    print("number of nodes in graph G: ",n)
    print("number of edges in graph G: ",m)
    print("number_connected_components in G: ",ncc)
    return(G)

def get_text_cluster_df_from_network(G):
    counter = 0
    cc_tuple = []
    for cc in nx.connected_components(G):
        counter+=1
        cc_tuple.append( (cc, len(cc)) )
        
    text_clusters_df = pd.DataFrame.from_records(cc_tuple)
    text_clusters_df.columns = ["unique_str_list", "num_unique_str"]
    text_clusters_df = text_clusters_df.sort_values(by="num_unique_str", ascending=False)    
    return(text_clusters_df)

def get_text_cluster_df_from_unique_str_list(unique_str_list):
    lsh, minhashes = get_minhashes_of_unique_str_list(unique_str_list)
    singelton_list, pair_comparisons_list = get_pair_comparisons_list(lsh, minhashes)
    G = get_text_cluster_network(singelton_list, pair_comparisons_list, unique_str_list)
    text_clusters_df = get_text_cluster_df_from_network(G)
    return(text_clusters_df)   


##############################################################
# Overlap score between two sentences
##############################################################  

def sentence_to_word_list(words):
    words = words.replace(",", " ")
    words_list = words.split()
    words_list = [word for word in words_list if len(word)>1]
    return(words_list)

def remove_spacy_stop_words(words):
    new_words = [word for word in words if not word in spacy_stop_words]
    return(new_words)

def get_lemma_string(words):
    sentence = sp(words)
    lemma_list = []
    for word in sentence:
        if len(word.lemma_) < len(word.text):
            lemma_list.append(word.lemma_)
        else:
            lemma_list.append(word.text)
    lemma_string = ' '.join(lemma_list)        
    return(lemma_string)


def clean_and_lemmatization_of_unique_words(words):
    words = words.lower()
    words_lemma = get_lemma_string(words)
    words_list = sentence_to_word_list(words_lemma)
    words_no_stop_list = remove_spacy_stop_words(words_list)
    words_no_stop_unique_list = list(set(words_no_stop_list))
    words_no_stop_str = ' '.join(words_no_stop_unique_list)
    return(words_no_stop_str)

def clean_and_compare_between_two_sentences(words1, words2):
    words1 = clean_and_lemmatization_of_unique_words(words1)
    words2 = clean_and_lemmatization_of_unique_words(words2)
    
    intersected_words_no_stop = set(words1.split()).intersection(set(words2.split()))
    overlap_score = len(intersected_words_no_stop)/ (len(words1.split()) + 
                                                     len(words2.split()) - len(intersected_words_no_stop))
    #overlap_score = len(intersected_words_no_stop)/min(len(words1.split()),
    #                                                   len(words2.split()))
    return(overlap_score)


##############################################################
# Utf-idf top keywords
############################################################## 

def get_filtered_pos_tokens(text):
    legal_pos = ["VERB", "NOUN", 'ADJ', 'ADV']
    custom_stop_list = ["go", "want"]
    doc = nlp(text)

    #collect_tokens = set()
    collect_tokens = []
    for token in doc:
        if not token.is_stop and token.pos_ in legal_pos:
            #print(token.text)
            #print(token.text, token.pos_,
                #token.lemma_, token.pos_, token.tag_, token.dep_,
                #  token.shape_, token.is_alpha, token.is_stop
            #     )
            #if token.text != token.lemma_:
            #  collect_tokens.add(token.text + " ("+ token.lemma_ + ")")
            #else:
            #collect_tokens.add(token.lemma_)
            #if (len(token.lemma_)) > 1 and token.lemma_ not in custom_stop_list:
            if (len(token.text)) > 2 and token.lemma_ not in custom_stop_list:
                #collect_tokens.append(token.lemma_)
                collect_tokens.append(token.text)

    ans = ' '.join(collect_tokens)    
    return(ans)

def unify_doc_query_with_doc_list(doc_list_for_tfidf, doc_to_query):
    doc_to_query = get_filtered_pos_tokens(doc_to_query)
    temp_doc_list = [get_filtered_pos_tokens(doc_to_query)]
    temp_doc_list.extend(doc_list_for_tfidf.copy())
    doc_list_for_tfidf = temp_doc_list.copy()
    return(doc_list_for_tfidf, doc_to_query)

def get_top_keywrods_with_tfidf_on_doc_itself(doc_list, ngram=2,):
    doc_list_for_tfidf = doc_list.copy()

    keyword_list = []
    for sentence in data:
        doc_to_query = sentence
        top_keywords = get_top_keywrods_with_tfidf(doc_list_for_tfidf, doc_to_query, ngram=2, num_return=10)
        temp_keyword_list = top_keywords.values.tolist()
        keyword_list.extend(temp_keyword_list)
        #print(doc_to_query)
        
    keyword_list_df = pd.DataFrame.from_records(keyword_list)
    #keyword_list_df
    keyword_list_df_sum = keyword_list_df.groupby(by=0).sum().reset_index()
    keyword_list_df_sum = keyword_list_df_sum.sort_values(by=1, ascending=False)
    keyword_list_df_sum.columns = ["keywords", "sum rank"]   
    
    return(keyword_list_df_sum[keyword_list_df_sum["rank"]>0])

def get_top_keywrods_with_tfidf(doc_list_for_tfidf, doc_to_query, ngram=2, num_return=10):
    # Unify the doc to query with the doc list
    doc_list_for_tfidf, doc_to_query = unify_doc_query_with_doc_list(doc_list_for_tfidf, doc_to_query)
    #print("doc_to_query:", doc_to_query)

    #tfidf = TfidfVectorizer(tokenizer=tokenize) # , stop_words='english'
    tfidf = TfidfVectorizer(ngram_range = (ngram,ngram))
    tfs = tfidf.fit_transform(doc_list_for_tfidf) # txt1

    #str = 'all great and precious things are lonely.'
    response = tfidf.transform([doc_to_query])  # txt1[0] # doc_list_for_tfidf[0]
    #print(response)

    feature_names = tfidf.get_feature_names()
    #for col in response.nonzero()[1]:
    #    print(feature_names[col], ' - ', response[0, col])

    # Getting top ranking features
    sums = response.sum(axis = 0)
    data1 = []
    for col, term in enumerate(feature_names):
        data1.append( (term, sums[0,col] ))
    ranking = pd.DataFrame(data1, columns = ['term','rank'])
    words = (ranking.sort_values('rank', ascending = False))
    #print ("\n\nWords head : \n", words.head(num_return)) 
    return(words[words["rank"]>0])

##############################################################
# Utils
##############################################################  

def find_all_occurrences_of_sub_in_string(substring, fullstring):
    ans = [m.start() for m in re.finditer(substring, fullstring)]
    return(ans)
    





