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
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from streamlit_chat import message
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
claude_params = {'anthropic_api_key':yaml_data["LLMS"]["CLAUDE_API_KEY"],
                'model':'claude-instant-v1.1-100k','max_tokens_to_sample':30000}
oai = OpenAIEmbeddings(openai_api_key = open_ai_params['openai_api_key'])
pinecone_db = Pinecone.from_existing_index('financial-analysis', oai)
#faiss_db = FAISS.load_local(folder_path = '../data/entiredocument',embeddings = oai)
llm = Cohere(**cohere_params)
llm = ChatAnthropic(**claude_params)
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
    #The code below gets the quota table from Yahoo Finance. The streamlit page
    #is divided into 2 columns and selected columns are displayed on each side of the page.
  
    #c1, c2 = st.columns((1,1))
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

def tab2():
   # PageBeginning()
    if ticker != '-':
            cache_count = 0
            ### If cache_count is 0, then initialize the entire query, else just input the final query because the chat messages already caches the data.         
            st.title('FinGPT 101')
            st.text('''The APIs under use in this application are strictly for academic purposes. The model under use over here is Claude 100k. 
            Claude's API key is limited to 1 concurrent user. If you find this application not working, it is 
                    probably because the server is loaded. Please try again after some time''')

            if 'generated' not in st.session_state:
                st.session_state['generated'] = []

            if 'past' not in st.session_state:
                st.session_state['past'] = []

            def get_text():
                input_text = st.text_input("You: ","", key="input",placeholder = "What do you want to know about {}".format(ticker))
                return input_text 


            user_input = get_text()
            @st.cache_data
            def getllmoutput(text,ticker,cache_count):
                stock_llm = LLM_analysis(ticker,open_ai_params,cohere_params,claude_params)
                if cache_count == 0:
                    query,file_name = stock_llm.qachain_anthropic(pinecone_db,text)
                else:
                    query = text
                messages = [HumanMessage(content=query)]
                outputquery = llm(messages).content
                cache_count = cache_count + 1
                df = stock_llm.process_file_names(file_name)
                if 'ticker' in df.columns:
                    df.drop(['ticker'],axis = 1,inplace = True)
                if 'final_score' in df.columns:
                    df.drop(['final_score'],axis = 1,inplace = True)
                    
                df.set_index(df.columns[0],inplace = True)    
                return outputquery,df


            if user_input:
                
                # output = query({
                #     "inputs": {
                #         "past_user_inputs": st.session_state.past,
                #         "generated_responses": st.session_state.generated,
                #         "text": user_input,
                #     },"parameters": {"repetition_penalty": 1.33},
                # })

                st.session_state.past.append(user_input)
                output_query,display_df = getllmoutput(user_input,ticker,cache_count)
                st.session_state.generated.append(output_query)

            if st.session_state['generated']:
                conversation_length = len(st.session_state['generated'])-1
                last_conversation = max(len(st.session_state['generated'])-5,-1)

                for i in range(conversation_length,last_conversation, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    st.text("Here are some more relevant information related to your question that you might be interested in:-")
                    st.dataframe(display_df)
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
             
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
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Ask me Anything'])
    if select_tab == 'Summary':
        tab1()
    elif select_tab == 'Ask me Anything':
        tab2()
    
if __name__ == "__main__":
    run()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    