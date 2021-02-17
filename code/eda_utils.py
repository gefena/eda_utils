
import numpy as np
from numpy import median
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import itertools
from networkx.algorithms import community
import networkx as nx

from pandas.plotting import autocorrelation_plot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from spacy.lang.en.stop_words import STOP_WORDS as spacy_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud

import time
import glob, os
import swifter
import missingno as msno
from IPython.display import display, Markdown

import ruptures as rpt
import changefinder

from datasketch import MinHash, MinHashLSH
from nltk import ngrams
import editdistance
import re

import multiprocessing
from multiprocessing import Pool


def infer_date_col(df, timezone_conversion=False):
    for col in df.columns:
        if (df[col].dtype == 'object') and (df[col].isnull().sum() != df[col].shape[0]):
            try:
                df[col] = pd.to_datetime(df[col])
                if timezone_conversion:
                    df[col] = df[col].dt.tz_convert(None)
                    print("col: " + col + " was transformed to date and timezone converted to UTC")

                print("col: " + col + " was transformed to date")
            #except ValueError:
            except:    
                pass
    return(df)

def get_percntage_missing_values(df):    
    num_rows = df.shape[0]
    for col in (df.columns):
        sum_missing = df[col].isnull().sum()
        print("col:" + col + ", missing values:", str(sum_missing/num_rows) + "%")
        
def missing_data(df):
    display(df.info())
    print()
    print("Percentage of missing data:")
    get_percntage_missing_values(df)
    msno.matrix(df)
    
def col_against_missing_data(df, compared_col):
    print("1 indicates missing values, 0 are non-missing")
    num_rows = df.shape[0]
    for col in (df.columns):
        sum_missing = df[col].isnull().sum()
        if sum_missing/num_rows > 0.1:
            missing_values_series = np.where(df[col].isnull(), 1, 0) # let's make a variable that indicates 1 if the observation was missing or zero otherwise
            df_tmp = pd.DataFrame()
            df_tmp[compared_col] = df[compared_col]
            new_col = col + "_missing_values"
            df_tmp[new_col] = missing_values_series
            print("medians: ",df_tmp.groupby(by=new_col)[compared_col].median())  
            fig, ax = plt.subplots(figsize=(20,6))
            ax = sns.violinplot(x=new_col, y=compared_col, data=df_tmp, scale='count', inner='box')
            #ax = sns.boxplot(x=col, y=compared_col, data=df_tmp)
            plt.title(compared_col + " grouped by " + new_col + ", median = " + str())
            plt.xticks(rotation=90)
            plt.show()


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
                #df[[col]].groupby(by=col).count().plot(kind="bar", title= str(col), figsize=(20,6))              
                df[col].value_counts().plot(kind="bar", title= str(col), figsize=(20,6))
                plt.show()
                df_tmp = df[col].value_counts() / df.shape[0]
                print("*** Rare categories:")
                display(df_tmp[df_tmp < 0.1])
            else:
                print("Too many (or just one) unique values for bar-plot")    
                
                
def eda_correlation_all_to_column(df, compared_col_list, min_max_corr_th = 0.3):
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
                df_tmp[new_col_name] = df_tmp[new_col_name].astype('str')
                df_tmp.groupby(by=new_col_name)[compared_col].median().plot(figsize=(20,2), title="Medians of " + compared_col + " over " + new_col_name)
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
                    df_tmp.groupby(by=col)[compared_col].median().plot(kind='bar', figsize=(20,2), title="Medians of " + compared_col + " over " + col)
                    plt.xticks(rotation=90)
                    plt.show() 
                if nunique >= 20 and nunique <= 30:
                    df_tmp = pd.DataFrame()
                    df_tmp[compared_col] = df[compared_col]
                    df_tmp[col] = df[col]
                    df_tmp.boxplot(by=col, figsize=(20,6))
                    plt.xticks(rotation=90)
                    plt.show()    
                    df_tmp.groupby(by=col)[compared_col].median().plot(kind='bar', figsize=(20,2), title="Medians of " + compared_col + " over " + col)
                    plt.xticks(rotation=90)
                    plt.show() 
            else:
                print("Too many (or just one) unique values for box-plot")
                

def eda_cols_vs_datetime_col(df, col_date, plotly_flag):                
    col_numerical = list(df.select_dtypes([np.number]).columns)
    #col_dates = list(df.select_dtypes(include=['datetime64']))
    for col in col_numerical:
        #print("column:", col, ", dtype:", df[col].dtype)
        #print("=======================================")
        display(Markdown("## " + "Column: " + col + ", dtype:" + str(df[col].dtype) ))
        print("count:", df[col].count())
        print("nunique:", df[col].nunique())
        print("isnull sum:",df[col].isnull().sum())
        print("zero count:",df[df[col]==0][col].count())
        #if col in col_numerical or col in col_dates:
        print("max:", df[col].max(), "min:", df[col].min())
        
        #fig = px.line(df, x=col_date, y=col, title="Timeseries for " + col)
        if plotly_flag:
            fig = go.Figure(go.Scatter(x = df[col_date], y = df[col]))        
            fig.update_layout(title_text="Timeseries for " + col)
            fig.update_xaxes(rangeslider_visible=True)
            fig.show()

            fig = px.histogram(df, x=col, nbins=20, title="Histogram for " +col)
            fig.show()
        else:
            #plt.figure(figsize=(20,3))
            #autocorrelation_plot(df[col])
            df.plot(x=col_date, y=col, title="Timeseries for " + col, figsize=(20,6))
            plt.show()
            
            fig, ax = plt.subplots(figsize=(14,6))
            df[col].hist(bins=50)
            plt.title("Histogram for " +col)
            plt.show()              
            
        display(Markdown("----"))

               
def eda_cols_vs_datetime_col_with_outliers(df, col_date, ma_window, one_step_ahead_flag, show_limits_flag):                
    col_numerical = list(df.select_dtypes([np.number]).columns)
    for col in col_numerical:
        display(Markdown("## " + "Column: " + col + ", dtype:" + str(df[col].dtype) ))
        print("count:", df[col].count())
        print("nunique:", df[col].nunique())
        print("isnull sum:",df[col].isnull().sum())
        print("zero count:",df[df[col]==0][col].count())
        print("max:", df[col].max(), "min:", df[col].min())
        
        date_series = df[col_date]
        series = df[col]
        series_name = col
        plot_time_series_with_outliers(date_series, series, series_name, one_step_ahead_flag, ma_window, show_limits_flag)
            
        #df.plot(x=col_date, y=col, title="Timeseries for " + col, figsize=(20,6))
        #plt.show()

        fig, ax = plt.subplots(figsize=(14,6))
        df[col].hist(bins=50)
        plt.title("Histogram for " +col)
        plt.show()              
            
        display(Markdown("----"))       
            
                   
def generate_column_correlation_network(df, th=0.8, edge_labels_flag=True, layout="spring_layout"):

    G = nx.Graph()
    col_numerical = list(df.select_dtypes([np.number]).columns)
    comb2_col_numerical = list(itertools.combinations(col_numerical, 2)) # Make combinations from col_numerical

    # go over all combinations calculate correlation
    for rec in comb2_col_numerical:
        col1 = rec[0]
        col2 = rec[1]
        corr = df[col1].corr(df[col2], method='pearson')
        corr = round(corr,2)
        if abs(corr) >= th:   # if correlation is high enoigh add edge to graph
            G.add_edge(col1, col2, weight = corr)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    if num_nodes <= 1 or num_edges == 0:
        print("netwotk has no nodes (or just 1) or edges")
        return
    
    # calculate communities
    if num_edges > 1:
        communities_generator = community.girvan_newman(G)
        top_level_communities = next(communities_generator)
        next_level_communities = next(communities_generator)

        display(Markdown("## Communites:") )
        node_to_cc_dict = dict()
        cc = sorted(map(sorted, next_level_communities))
        counter = 0
        if len(cc) == 0:
            print("netwotk has no communites")
            return

        for c in cc:
            counter+=1
            for n in list(c):
                node_to_cc_dict[n] = counter
            print(c)
            print("-------------------")  

    # print communites and plot
    display(Markdown("## Network:") )
    print("number of nodes:", num_nodes)
    print("number of edges:", num_edges)
    plt.figure(figsize=(26,10)) 
    if layout=="spring_layout":
        pos=nx.spring_layout(G, k=0.15, iterations=20, scale=2)
    elif layout=="planar_layout": 
        pos=nx.planar_layout(G)
    elif layout=="circular_layout":
        pos=nx.circular_layout(G)
    else:    
        print("problrm with choosing a layout")
        
    if num_edges > 1:
        com_values = [node_to_cc_dict[n] for n in G.nodes()]
        nx.draw_networkx(G,pos, cmap = plt.get_cmap('jet'), node_color = com_values, with_labels=True,  alpha=0.6, font_size=14)
    else:
        nx.draw_networkx(G,pos, cmap = plt.get_cmap('jet'), with_labels=True,  alpha=0.6, font_size=16)
    if edge_labels_flag:
        edge_labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,  alpha=0.8)
    plt.show()   
        
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


def fix_limits_for_time_series(series_quantile_array):
    # if fit is skewd so the entire boxplot is above/below zero we fix it to be symetric
    if max(series_quantile_array) < 0:
        series_quantile_array[0] = min(series_quantile_array)
        series_quantile_array[1] = abs(series_quantile_array[0]) #0.5*iqr
    if min(series_quantile_array) > 0:
        series_quantile_array[0] = -1*max(series_quantile_array) #-0.5*iqr
        series_quantile_array[1] = max(series_quantile_array)
    return(series_quantile_array)    

def get_outlier_limits_iqr(series, chosen_quantiles=[0.25,0.75], flag_fix_limits_for_time_series=False):
    series_quantile = series.quantile(chosen_quantiles)
    series_quantile_array = series_quantile.values
    if flag_fix_limits_for_time_series:
        series_quantile_array = fix_limits_for_time_series(series_quantile_array)        
    iqr = max(series_quantile_array) - min(series_quantile_array)
    limits = series_quantile_array + (1.5*iqr) * np.array([-1,1])
    return(limits)

def remove_max_outliers(series, chosen_quantiles=[0.25,0.75]):
    limits = get_outlier_limits_iqr(series, chosen_quantiles)
    series[series > max(limits)] = np.nan
    return(series)

def get_outlier_limits_normal_dist(series):
    std_relation = series.std()
    mean_relation = series.mean()
    limits = (mean_relation - 3*std_relation, mean_relation + 3*std_relation)
    return(limits)

def get_df_with_normalzone_limits_from_reiduals(fitted, resid, learn_period_len, sensitivity = "less"):
    flag_fix_limits_for_time_series = True
    if(sensitivity=="less"):
        limits = get_outlier_limits_iqr(resid[0:learn_period_len], [0.1,0.9], flag_fix_limits_for_time_series)
    else:
        limits = get_outlier_limits_iqr(resid[0:learn_period_len], [0.25,0.75], flag_fix_limits_for_time_series)

    dfWithLimits = pd.DataFrame()
    dfWithLimits["lowerLimit"] = fitted + limits[0]
    dfWithLimits["upperLimit"] = fitted + limits[1]
    return (dfWithLimits)

def plot_time_series_with_outliers(date_series, series, series_name, one_step_ahead_flag, ma_window, show_limits_flag):
    
    if one_step_ahead_flag:
        ma_series = series.ffill().rolling(ma_window).mean().shift(1) # use moving avaerage as a forecasting mechanism
    else:
        ma_series = series.ffill().rolling(ma_window).mean()

    residual_series = ma_series - series
    learn_period_len = series.shape[0] # Learning period can be shorter than entire series for some applicaions
    sensitivity = "regular"
    df_limits = get_df_with_normalzone_limits_from_reiduals(ma_series, residual_series, learn_period_len, sensitivity)

    df_limits["value"] = series
    df_limits["residuals"] = residual_series
    df_limits["outliers"] = None
    # Mark each outlier in series
    df_limits.loc[(df_limits["value"] - df_limits["upperLimit"])>0, "outliers"] = \
                                            df_limits[(df_limits["value"] - df_limits["upperLimit"])>0]["value"]
    df_limits.loc[(df_limits["lowerLimit"] - df_limits["value"])>0, "outliers"] = \
                                            df_limits[(df_limits["lowerLimit"] - df_limits["value"])>0]["value"]

    df_with_limits_and_outliers = pd.DataFrame(date_series)
    df_with_limits_and_outliers.columns = ["date"]

    df_with_limits_and_outliers = pd.concat([df_with_limits_and_outliers, df_limits], axis=1)
    percentage_of_outliers = (~df_with_limits_and_outliers["outliers"].isnull()).sum() / df_with_limits_and_outliers.shape[0]
    print("percentage_of_outliers:", percentage_of_outliers)

    fig, ax = plt.subplots(figsize=(22,8))
    df_with_limits_and_outliers.plot(x="date", y="value" , ax=ax, c='blue', grid=True)
    if df_with_limits_and_outliers["outliers"].isnull().sum() < df_with_limits_and_outliers.shape[0]:
        df_with_limits_and_outliers.plot(x="date", y="outliers" , ax=ax, marker="o", c='red', grid=True, title="Timeseries for " + series_name)
        if show_limits_flag:
            df_with_limits_and_outliers.plot(x="date", y="upperLimit" , ax=ax, c='green', grid=True)
            df_with_limits_and_outliers.plot(x="date", y="lowerLimit" , ax=ax, c='green', grid=True)
        plt.show()  
        
##############################################################
# changepoints / breakpoints in time series
##############################################################

def find_changepoints_for_time_series(series, modeltype="binary", number_breakpoints=10, plot_flag=True, plot_with_dates=False, show_time_flag=False):
    
    #RUPTURES PACKAGE
    #points=np.array(series)
    points=series.values
    title=""
    
    t0 = time.time()
    if modeltype=="binary":
        title="Change Point Detection: Binary Segmentation Search Method"
        model="l2"
        changepoint_model = rpt.Binseg(model=model).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)
    if modeltype=="pelt":
        title="Change Point Detection: Pelt Search Method"
        model="rbf"
        changepoint_model = rpt.Pelt(model=model).fit(points)
        result = changepoint_model.predict(pen=10)    
    if modeltype=="window":
        title="Change Point Detection: Window-Based Search Method"
        model="l2"
        changepoint_model = rpt.Window(width=40, model=model).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)  
    if modeltype=="Dynamic":
        title="Change Point Detection: Dynamic Programming Search Method"
        model="l1"
        changepoint_model = rpt.Dynp(model=model, min_size=3, jump=5).fit(points)
        result = changepoint_model.predict(n_bkps=number_breakpoints)
    if modeltype=="online":
        # CHANGEFINDER PACKAGE
        title="Simulates the working of finding changepoints in online fashion"
        cf = changefinder.ChangeFinder()
        scores = [cf.update(p) for p in points]
        result = (-np.array(scores)).argsort()[:number_breakpoints]
        result = sorted(list(result))
        if series.shape[0] not in result:
            result.append(series.shape[0])
        

    if show_time_flag:
        elapsed_time = time.time() - t0
        print("[exp msg] elapsed time for process: " + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))) 
    
    if plot_flag:
        if not plot_with_dates:
            rpt.display(points, result, figsize=(18, 6))
            plt.title(title)
            plt.show()
        else:    
            series.plot(figsize=(18, 6))
            plt.title(title)
            for i in range(len(result)-1):  
                if i%2==0:
                    current_color='xkcd:salmon'
                else:
                    current_color='xkcd:sky blue'        
                #plt.fill_between(series.index[result[i]:result[i+1]], series.max(), color=current_color, alpha=0.3)
                plt.fill_between(series.index[result[i]:result[i+1]], y1=series.max()*1.1, y2=series.min()*0.9, color=current_color, alpha=0.3)
            plt.show()    
                
    return(result)  

        
##############################################################
# Explore file/data problems
##############################################################


def fix_data_file(origfile, newfile):

    with open(origfile) as f:
        lines = f.readlines()
        print("number of lines:", len(lines))

        with open(newfile, "w") as w:

            counter = 0
            for line in lines:
                counter+=1
                new_line = re.sub(' +,', ',', line)
                new_line = re.sub(', +', ',', new_line)
                new_line = re.sub('NAN', '', new_line)
                w.write(new_line)
                
    print("Finished writing newfile")           
                
                
def is_column_mixed_type(df, col):
    ''' Find if a column has mixed data types: float / int / str '''
    num_float = df[df[col].apply(lambda x: isinstance(x, float))].shape[0]
    num_str = df[df[col].apply(lambda x: isinstance(x, str))].shape[0]
    num_int = df[df[col].apply(lambda x: isinstance(x, int))].shape[0]
    if num_float*num_str > 0 or num_float*num_int > 0 or num_str*num_int > 0:
        print("col:", col, "- has mixed types - ", "num_str:", num_str, ", num_float:", num_float, ", num_int:", num_int)
        print("uniques:")
        print(list(df[col].unique())[:10])
        print("================================================")
        return(True)
    else:
        return(False)  

    
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
# Given a dictionary of dataframes and action over dataframe retrun answer
############################################################## 

def get_dict_of_df_subgroups(df, subgroup_col, copy_data=False):
    dict_df = dict()
    subgroup_list = df[subgroup_col].unique()
    for subgroup in subgroup_list:
        if copy_data:
            dict_df[subgroup] = df[df[subgroup_col]==subgroup].copy()
        else:
            dict_df[subgroup] = df[df[subgroup_col]==subgroup]
            
    return(dict_df)  

def worker(subgroup, subgroup_df, return_dict, methodtorun):
    '''worker function'''
    return_dict[subgroup] = methodtorun(subgroup_df)

def multiprocess_function_on_subgroup_df(subgroup_list, dict_df, methodtorun):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    for i in range(len(subgroup_list)):
        subgroup = subgroup_list[i]
        subgroup_df = dict_df[subgroup]
        p = multiprocessing.Process(target=worker, args=(subgroup, subgroup_df, return_dict, methodtorun) )
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    #print(return_dict.values())
    
    ans_df = pd.DataFrame.from_records(return_dict.items())
    ans_df.columns = ["subgroup", "subgroup_ans"]
    
    return(ans_df)

def multiprocess_pool_function_on_subgroup_df(subgroup_list, dict_df, methodtorun):
    input_list_for_pool = []
    for i in range(len(subgroup_list)):
        subgroup = subgroup_list[i]
        subgroup_df = dict_df[subgroup]
        input_for_pool = (subgroup, subgroup_df)
        input_list_for_pool.append(input_for_pool)

    p = Pool(6)
    return_list = p.map(methodtorun, input_list_for_pool)    
    
    ans_df = pd.DataFrame.from_records(return_list)
    ans_df.columns = ["subgroup", "subgroup_ans"]
   
    return(ans_df) 

def function_on_subgroup_df(subgroup_list, dict_df, methodtorun):
    return_dict = dict()
    for i in range(len(subgroup_list)):
        subgroup = subgroup_list[i]
        subgroup_df = dict_df[subgroup]
        return_dict[subgroup] = methodtorun(subgroup_df)
        
    ans_df = pd.DataFrame.from_records(list(return_dict.items()))
    ans_df.columns = ["subgroup", "subgroup_ans"]
    return(ans_df) 


##############################################################
# Correlations and predictive power scores
# - https://8080labs.com/blog/posts/rip-correlation-introducing-the-predictive-power-score-pps/
# - https://github.com/8080labs/ppscore
##############################################################    

def plot_corrmatrix(corrMatrix, title):
    fig = plt.figure(figsize=(12, 10))

    hit = sns.heatmap(corrMatrix, annot=True, cmap= 'coolwarm') # cbar_kws= {'orientation': 'horizontal'}     
    hit.set_xticklabels(corrMatrix.columns, fontsize=14, rotation=45)
    hit.set_yticklabels(corrMatrix.columns, fontsize=14)
    #plt.xticks(range(corrMatrix.shape[1]), corrMatrix.columns, fontsize=14, rotation=45)
    #plt.yticks(range(corrMatrix.shape[1]), corrMatrix.columns, fontsize=14)
    plt.title(title, fontsize=16);
    plt.show()

# For feature importance based on pps - go over all features compared to the target    
# Predictive power score matrix
#import ppscore as pps
#fearue_col = "neighbourhood_group"
#target_col = "price"
#pps.score(data, fearue_col, target_col)    