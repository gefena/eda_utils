
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud

#import time
#import glob, os
import swifter
import missingno as msno
from IPython.core.display import display


def infer_date_col(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                print("col: " + col + " was transformed to date")
            except ValueError:
                pass
    return(df)
        
def missing_data(df):
    display(df.info())
    msno.matrix(df)
    
def eda_basic(df):
    col_numerical = list(df.select_dtypes([np.number]).columns)
    col_dates = list(df.select_dtypes(include=['datetime64']))
    col_objects = list(df.select_dtypes(include=['object']))

    for col in list(df.columns):
        print()
        print("column:", col, ", dtype:", df[col].dtype)
        print("=======================================")
        print("count:", df[col].count())
        print("nunique:", df[col].nunique())
        print("isnull sum:",df[col].isnull().sum())
        print("zero count:",df[df[col]==0][col].count())
        if col in col_numerical or col in col_dates:
            print("max:", df[col].max(), "min:", df[col].min())
        print("top value counts:")
        temp_df = pd.DataFrame(df[col].value_counts()).head(3)
        temp_df.columns = ["count"]
        temp_df[col] = temp_df.index
        temp_df = temp_df[[col,"count"]]
        print(temp_df.to_string(index=False))
        if col in col_numerical and df[col].nunique() > 1:
            fig, ax = plt.subplots(figsize=(14,6))
            df[col].hist(bins=50)
            plt.title(col)
            plt.show()    
        if col in col_dates:
            series_tmp = df[col].swifter.apply(lambda x: 100*x.year + x.month);
            series_tmp.groupby(series_tmp).count().plot(kind="bar", title= "Monthly count of " + str(col), figsize=(20,6))
            plt.show() 

        if col in col_objects:
            nunique = df[col].nunique()
            print("nunique:", nunique)
            if nunique > 1 and nunique < 50:
                df[[col]].groupby(df[col]).count().plot(kind="bar", title= str(col), figsize=(20,6))
                plt.show()
            else:
                print("Too many (or just one) unique values for bar-plot")    
                
                
def eda_correlation_all_to_column(df, compared_col_list, min_max_corr_th = 0.3):
    #
    # Check if compared_col_list is numerical
    #
    
    col_numerical = list(df.select_dtypes([np.number]).columns)
    col_dates = list(df.select_dtypes(include=['datetime64']))
    col_objects = list(df.select_dtypes(include=['object']))
    
    if not isinstance(compared_col_list, list):
        compared_col_list = [compared_col_list]
    
    compared_col_list_copy = compared_col_list.copy()
    compared_col_list = []
    for compared_col in compared_col_list_copy:
        if compared_col in col_numerical:
            compared_col_list.append(compared_col)
        else:
            print("col: " + compared_col + " is not numerical")    

    if len(compared_col_list) == 0:
        return("No numerical columns were given")
    
    for col in list(df.columns):
        print()
        print("column:", col, ", dtype:", df[col].dtype)
        print("=======================================")
        for compared_col in compared_col_list:
            print("compared_col:", compared_col)
            if col in col_numerical and df[col].nunique() > 1 and col!=compared_col:
                corr_pearson = df[compared_col].corr(df[col], method='pearson')
                print("corr_pearson: ", corr_pearson)
                if corr_pearson > min_max_corr_th or corr_pearson < -1*min_max_corr_th:
                    df.plot.scatter(x=compared_col, y = col, figsize=(10,6))
                    plt.title(col + " .vs. " + compared_col)
                    plt.show()    
            if col in col_dates:
                series_tmp = df[col].swifter.apply(lambda x: 100*x.year + x.month);
                corr_pearson = df[compared_col].corr(series_tmp, method='pearson')
                print("corr_pearson: ", corr_pearson)
                df_tmp = pd.DataFrame()
                df_tmp[compared_col] = df[compared_col]
                new_col_name = col+"_month"
                df_tmp[new_col_name] = series_tmp
                df_tmp.boxplot(by=new_col_name, figsize=(20,6))
                plt.xticks(rotation=90)
                plt.show()      
            if col in col_objects:
                nunique = df[col].nunique()
                print("nunique:", nunique)
                if nunique > 1 and nunique < 20:
                    df_tmp = pd.DataFrame()
                    df_tmp[compared_col] = df[compared_col]
                    df_tmp[col] = df[col]                
                    fig, ax = plt.subplots(figsize=(20,6))
                    ax = sns.violinplot(x=col, y=compared_col, data=df_tmp, scale='count', inner='box')
                    plt.title(compared_col + " grouped by " + col)
                    plt.xticks(rotation=90)
                    plt.show()
                if nunique >= 20 and nunique <= 30:
                    df_tmp = pd.DataFrame()
                    df_tmp[compared_col] = df[compared_col]
                    df_tmp[col] = df[col]
                    df_tmp.boxplot(by=col, figsize=(20,6))
                    plt.xticks(rotation=90)
                    plt.show()     
            else:
                print("Too many (or just one) unique values for box-plot")
                
                
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
    
    
def get_outlier_limits_iqr(series):
    series_quantile = series.quantile([0.25,0.75])
    series_quantile_array = series_quantile.values
    iqr = max(series_quantile_array) - min(series_quantile_array)
    limits = series_quantile_array + (1.5*iqr) * np.array([-1,1])
    return(limits)

def get_outlier_limits_normal_dist(series):
    std_relation = series.std()
    mean_relation = series.mean()
    limits = (mean_relation - 3*std_relation, mean_relation + 3*std_relation)
    return(limits)    
 
def get_hist_before_after_outlier_removal(df, col):
    series = df[col]
    limit_edges = get_outlier_limits_iqr(series)
    print("Limit_edges: ", limit_edges)
    series_no_outliers = series[(series >= limit_edges[0]) & (series <= limit_edges[1])]                
    fig, ax = plt.subplots(figsize=(14,6))
    series.hist(bins=50)
    plt.title(col + " before outliers removal")
    plt.show()                  
    fig, ax = plt.subplots(figsize=(14,6))
    series_no_outliers.hist(bins=50)
    plt.title(col + " after outliers removal")
    plt.show() 
