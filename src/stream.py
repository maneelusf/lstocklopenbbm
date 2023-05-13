import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import yaml
from langchain.vectorstores import FAISS,Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import Cohere, OpenAI, AI21
from llm import LLM_analysis
import pinecone


with open('../data/apis.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
pinecone.init(api_key= yaml_data['PINECONE']['API_KEY'],environment= yaml_data['PINECONE']['ENV'])
index_name = "financial-analysis"
open_ai_params = {'max_tokens':1000,'openai_api_key' : yaml_data['LLMS']['OPENAI_API_KEY'],'temperature' :0,'model_name':'text-davinci-003'}
cohere_params = {
    "model": "command-xlarge-nightly",
    "max_tokens": 1000,
    "cohere_api_key": yaml_data["LLMS"]["COHERE_API_KEY"],
    "temperature": 0,
    "k": 0,
}
ai21_params = {
    "model": "j2-jumbo-grande",
    "numResults": 1,
    "temperature": 0,
    "topP": 1,
    "ai21_api_key": yaml_data["LLMS"]["AI21_API_KEY"],
    "maxTokens": 2000,
}
oai = OpenAIEmbeddings(openai_api_key = open_ai_params['openai_api_key'])
pinecone_db = Pinecone.from_existing_index('financial-analysis', oai)
#faiss_db = FAISS.load_local(folder_path = '../data/entiredocument',embeddings = oai)
llm = Cohere(**cohere_params)
#import pyfolio as pf

#==============================================================================
# Tab 1 Summary
#==============================================================================

def tab1():
    #c1,c2 = st.columns((1,3))
    @st.cache_data
    def getsummary(ticker):
            table = si.get_quote_table(ticker, dict_result = False)
            return table 
    def getstockdata(ticker):
        stockdata = yf.download(ticker, period = 'MAX')
        return stockdata
    @st.cache_data
    def getllmoutput(text,ticker):
        stock_llm = LLM_analysis(ticker,open_ai_params,cohere_params,ai21_params)
        query,file_name = stock_llm.qachain(pinecone_db,text)
        outputquery = llm(query)
        outputquery = outputquery.replace('$',"\\$")
        df = stock_llm.process_file_names(file_name)
        if df.shape != (0,0):
            df.drop(['ticker'], axis=1, errors='ignore',inplace = True)
        return outputquery,df


         
    
          
          
    
    
    #The code below gets the quota table from Yahoo Finance. The streamlit page
    #is divided into 2 columns and selected columns are displayed on each side of the page.
  
    c1, c2 = st.columns((1,1))
    
    with c2:
        if ticker != '-':        
            st.title('Stock Performance')
            chartdata = getstockdata(ticker)                    
            fig = px.area(chartdata, chartdata.index, chartdata['Close'])
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label = "MAX", step="all")
                    ])
                )
            )
            st.plotly_chart(fig)
            summary = getsummary(ticker)
            summary['value'] = summary['value'].astype(str)
            c3, c4 = st.columns((1,1))
            with c3:
                  
                showsummary_v1 = summary.iloc[[11, 4, 13, 7, 8, 10, 9, 0],]
                showsummary_v1.set_index('attribute', inplace=True)
                st.dataframe(showsummary_v1)
            with c4:
                showsummary = summary.iloc[[14, 12, 5, 2, 6, 1, 16, 3],]
                showsummary.set_index('attribute', inplace=True)
                st.dataframe(showsummary)

    with c1:
        if ticker == '-':
            st.title("FinGPT 101")
            text = st.text_input("Please select a ticker")        
        if ticker != '-':
            st.title("FinGPT 101")
            text = st.text_input('What do you want to know about {}?'.format(ticker),placeholder = "Enter your text")
            if text == '':
                pass     
            else:
                outputquery,df = getllmoutput(text,ticker)
                st.write(outputquery)
                st.dataframe(df)
    
    if ticker != '-':
        st.header("Sentiment Analysis")
        color = st.select_slider('Select a sentiment type',
    options=['Strongly Positive', 'Positive', 'Neutral', 'Negative', 'Strongly Negative'])
        df = pd.read_csv('../data/output/sentiment_scores.csv')
        df = df[(df['ticker'] == ticker) & (df['similarity'] == color)][['headline','datetime','similarity']]
        df.reset_index(inplace = True,drop = True)
        df.columns = ['Headline','Datetime','Sentiment']
        hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        st.dataframe(df)
         
    


             
    #The code below uses the yahoofinance package to get all the available stock
    #price data. Plotly is then used to visualize the data.  An interesting feature
    #from plotly called range selector is also used. A list of dictionaries
    #is added to range selector to make buttons and identify the periods.
    #References:
    #https://plotly.com/python/range-slider/
    
        
        
    
#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    st.set_page_config(layout="wide")
    ticker_list = os.listdir('../ticker')
    ticker_list = sorted(ticker_list)
    ticker_list = [ticker for ticker in ticker_list if ticker!='.DS_Store']
    ticker_list.insert(0,'-')
    
    
    #
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    tab1()
    
if __name__ == "__main__":
    run()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    