import yaml
import yahoo_fin.stock_info as si
import yfinance as yf
from langchain.llms import Cohere, OpenAI
from langchain.chains import SequentialChain,AnalyzeDocumentChain
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from langchain.chat_models import ChatAnthropic
from langchain.chains.summarize import load_summarize_chain
import os
from datetime import datetime
from langchain.vectorstores import FAISS,Pinecone
from langchain.embeddings import OpenAIEmbeddings
from openbb_terminal.sdk import openbb
with open("../data/apis.yaml", "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
open_ai_params = {
    "max_tokens": 512,
    "openai_api_key": yaml_data["LLMS"]["OPENAI_API_KEY"],
}
cohere_params = {
    "model": "command-xlarge-nightly",
    "max_tokens": 2202,
    "cohere_api_key": yaml_data["LLMS"]["COHERE_API_KEY"],
    "temperature": 0,
    "k": 0,
}

claude_params = {'anthropic_api_key':yaml_data["LLMS"]["CLAUDE_API_KEY"],
                'model':'claude-instant-v1.1-100k','max_tokens_to_sample':30000}

class StockLLM:
    def __init__(self, ticker):
        self.ticker = ticker

    def stock_availability(self):
        return self.ticker in os.listdir("./ticker")

    def sec_analysis_agent(self):
        ### Get sec files
        file_path = f"../ticker/{self.ticker}/fa/analysis_sec.txt"
        try:
            with open(file_path, "r") as f:
                file = f.read()
        except:
            raise Exception("This file is unavailable")
        return file

class LLM_analysis:
    def __init__(self, ticker, open_ai_params, cohere_params,claude_params):
        ### Requires both Cohere/OpenAI/Claude APIs
        self.ticker = ticker
        self.open_ai_params = open_ai_params
        self.cohere_params = cohere_params
        self.claude_params = claude_params
        self.cohere_llm = Cohere(**self.cohere_params)
        self.open_ai_llm = OpenAI(**self.open_ai_params)
        self.claude_llm = ChatAnthropic(**claude_params)
        self.stockllm = StockLLM(self.ticker)
  
    def query_user(self,file,type_of_file):
        if type_of_file not in ['pdf','txt','link','csv']:
            raise NotImplementedError("This file extension has not been implemented.")
        if type_of_file == 'pdf':
            pages = [page.extract_text() for page in pypdf.PdfReader(file).pages]
            text = '\n'.join(pages)

        if type_of_file in ['txt','csv']:
            with open(file,'r') as f:
                text = f.read()
        
        if type_of_file == 'link':
            loader = SeleniumURLLoader(urls=[file])
            data = loader.load()
            text = data[0].page_content
        llm = Cohere(temperature=0,cohere_api_key = cohere_params["cohere_api_key"])
        summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
        summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
        summary = summarize_document_chain.run(text)
        final_class = self.input_from_user_embedding_shot(summary)
        return final_class
    
    def context_precursor(self):
        if self.ticker is not None:
            entire_context = []
            for file in sorted(os.listdir('../ticker/{}/fa'.format(self.ticker))):
                if file == 'analysis_sec.txt':
                    with open('../ticker/{}/fa/analysis_sec.txt'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the summary of the latest SEC filings figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'est.csv':
                    with open('../ticker/{}/fa/est.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are some analyst estimates from Business Insider\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'income.csv':
                    with open('../ticker/{}/fa/income.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the income statement figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'balance.csv':
                    with open('../ticker/{}/fa/balance.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the balance sheet statement figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'cash.csv':
                    with open('../ticker/{}/fa/cash.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the cash flow statement figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'ratios.csv':
                    with open('../ticker/{}/fa/ratios.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the financial ratios statement figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)
                elif file == 'fraud.csv':
                    with open('../ticker/{}/fa/fraud.csv'.format(self.ticker),'r') as f:
                        x = f.read()
                        context_precursor = '''Here are the fraud ratios statement figures\n'''
                        final_context = context_precursor + x
                        entire_context.append(final_context)

            if os.path.exists('../ticker/{}/news/c_news.csv'.format(self.ticker)):
                x = pd.read_csv('../ticker/{}/news/c_news.csv'.format(self.ticker))
                x['datetime'] = pd.to_datetime(x['datetime'])
                datetimemax = x['datetime'].max()
                datetimemin = datetimemax + relativedelta(days = -7)
                x = '\n'.join(list(x[(x['datetime'] >= datetimemin) & (x['datetime'] <= datetimemax)].head(10)['headline']))
                context_precursor = '''Here are some recent news\n'''
                final_context = context_precursor + x
                entire_context.append(final_context)
            #### last 7 days stock information
            current_information = '''Latest stock price\n. Today's date is {} and the current stock price is {}'''.format(\
                str(datetime.today()),si.get_live_price(self.ticker))
            entire_context.append(current_information)
            
            ### Adding today's date and last 7 days stock movement  
            def current_info(ticker):
                x = yf.download(ticker)[['Close', 'Volume']]
                x = x.asfreq('D')
                last_7_days = x.dropna().tail(7)
                last_12_months = x.resample('M').last().tail(12)
                last_index = last_12_months.index[-1]
                new_last_index = datetime.today()
                new_index = list(last_12_months.index)
                new_index[-1] = new_last_index
                new_index = [str(x.date()) for x in new_index]
                last_12_months.index = new_index
                return last_7_days,last_12_months
            last_7_days,last_12_months = current_info(self.ticker)
            last_7_days_information = '''Last 7 days stock price(Close) and Volume\n {}'''.format(last_7_days.to_string())
            last_12_months_information = '''Last 12 months stock price(Close) and Volume\n {}'''.format(last_12_months.to_string())
            entire_context.extend([last_7_days_information,last_12_months_information])
            entire_context = '\n\n'.join(entire_context)
            self.entire_context = entire_context
            
    def qachain_anthropic(self,vectorstore,query):
        if self.ticker == None:
            raise Exception("Ticker must be present")
        else:
            self.context_precursor()
            filter_dict = {'$and':[{'ticker':self.ticker},{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
        k_count = max(len(set([doc.metadata['file_path'] for doc in documents])),3)*5
        if k_count != 5:
            documents = vectorstore.as_retriever(search_kwargs={"k": k_count,'filter':filter_dict}).get_relevant_documents(query)

        file_names = [doc.metadata['file_path'] for doc in documents]

        prompt = '''Use the following information to answer the question at the end in a coherent summary. 
    The below contains information about {ticker} and you are a financial analyst
    {context_precursor}
    Question: {question}
    Do not mention anything in your response about the context/information'''.format(ticker = self.ticker,context_precursor = self.entire_context,question = query )
        return prompt,file_names
    def qachain(self,vectorstore,query):
        if self.ticker == None:
            filter_dict = {'$and':[{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
        else:
            filter_dict = {'$and':[{'ticker':self.ticker},{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
        k_count = max(len(set([doc.metadata['file_path'] for doc in documents])),3)*5
        if k_count != 5:
            documents = vectorstore.as_retriever(search_kwargs={"k": k_count,'filter':filter_dict}).get_relevant_documents(query)
        page_content = '\n\n'.join([doc.page_content for doc in documents])
        file_names = [doc.metadata['file_path'] for doc in documents]
        meta_data = documents[0].metadata
       # file_path = 
        context_precursor =  '''The below contains information about {} and you are a financial analyst'''.format(meta_data['ticker'])
        prompt_template = """Use the following information to answer the question at the end in a coherent summary. 
    {context_precursor}
    {page_content}
    Question: {question}
    Think step by step.Do not mention "As per the information or context" in your response.
    """
        prompt = prompt_template.format(context_precursor = context_precursor,page_content = page_content,question = query)
        context_full_doc = []
        return prompt,documents
    
    def qachain_anthropic(self,vectorstore,query):
        if self.ticker == None:
            raise Exception("Ticker must be present")
        else:
            self.context_precursor()
            filter_dict = {'$and':[{'ticker':self.ticker},{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
        k_count = max(len(set([doc.metadata['file_path'] for doc in documents])),3)*5
        if k_count != 5:
            documents = vectorstore.as_retriever(search_kwargs={"k": k_count,'filter':filter_dict}).get_relevant_documents(query)

        file_names = [doc.metadata['file_path'] for doc in documents]

        prompt = '''Use the following information to answer the question at the end in a coherent summary. 
    The below contains information about {ticker} and you are a financial analyst
    {context_precursor}
    Question: {question}
    Do not mention anything in your response about the context/information'''.format(ticker = self.ticker,context_precursor = self.entire_context,question = query )
        return prompt,file_names
    
    def qachain_comparision(self,ticker_list,query):
        question_check = ["This question is related to the balance sheets of the stocks",\
 "This question is related to the share price movement of the stocks",\
    "This question is related to the cashflow of the stocks",\
      "This question is related to the income statement of the stocks",\
       "This question is related to the volume of the stocks"]
        oai = OpenAIEmbeddings(openai_api_key = open_ai_params['openai_api_key'])
        faiss_query = FAISS.from_texts(question_check,oai)
        x = faiss_query.similarity_search_with_relevance_scores(query)
        dictionary = {"This question is related to the balance sheets of the stocks":'Balance Sheet',
                    "This question is related to the share price movement of the stocks":"Share Price",\
            "This question is related to the cashflow of the stocks":"Cash Flow",\
            "This question is related to the income statement of the stocks":"Income Statement",\
            "This question is related to the volume of the stocks":"Volume"}
        similarities = [y[1] for y in x]
        sub = [similarities[x] - similarities[0] if x!=0 else 0 for x in range(0,len(similarities))]
        sub = sum([0 if abs(x)>0.025 else 1 for x in sub])
        x = x[:sub]
        page_content = [y[0].page_content for y in x]
        context = []
        for content in page_content:
            if dictionary[content] == 'Balance Sheet':
                content = '''This is the balance sheet information related to these stocks\n{}'''.format(openbb.stocks.ca.balance(ticker_list).to_string())
                context.append(content)
            elif dictionary[content] == 'Share Price':
                content = '''This is the share price information related to these stocks\n{}'''.format(openbb.stocks.ca.hist(ticker_list).to_string())
                context.append(content)
            elif dictionary[content] == 'Cash Flow':
                content = '''This is the cash flow information related to these stocks\n{}'''.format(openbb.stocks.ca.cashflow(ticker_list).to_string())
                context.append(content)
            elif dictionary[content] == 'Income Statement':
                content = '''This is the income statement information related to these stocks\n{}'''.format(openbb.stocks.ca.income(ticker_list).to_string())
                context.append(content)
            elif dictionary[content] == 'Volume':
                content = '''This is the volume information related to these stocks\n{}'''.format(openbb.stocks.ca.volume(ticker_list).to_string())
                context.append(content)
        final_context = '''You are a financial analyst. Looking at the above information, answer the following question.\n
        If the context information is not related, just say "I don't know"
        Question: {}'''.format(query)
        context.append(final_context)
        context = '''\n\n'''.join(context)
        return context
    
    def process_file_names(self,file_names):
        csv_filter = [file_name for file_name in file_names if '.csv' in file_name]
        df = pd.read_csv(csv_filter[0])
        df.rename(columns = {'Unnamed: 0':'Description'},inplace = True)
        return df       
