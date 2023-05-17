import yaml
from langchain.llms import Cohere, OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains import SequentialChain,AnalyzeDocumentChain
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fuzzywuzzy import fuzz, process
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.embeddings import CohereEmbeddings,OpenAIEmbeddings
from langchain.vectorstores import FAISS
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
from langchain.document_loaders import DataFrameLoader,CSVLoader,SeleniumURLLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import pypdf
import os

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

    def sec_chain_analysis(self):
        ### Initally we need a good bullet point summary of the latest sec filings

        template = """
"This is the sec summary of {stock}.\n
{summary}\n"
Can you summarize the text into bullet points with numbers in detail. Be as detailed as possible:-
"""
        sec_template = PromptTemplate(
            template=template, input_variables=["stock", "summary"]
        )
        sec_chain = LLMChain(
            llm=self.cohere_llm, prompt=sec_template, output_key="sec_summary"
        )
        template = """You are a financial analyst. Based on the below bullet points, can you further separate them into positive
and negative news in bullet points. Please do not leave out any point and go step by step.
{sec_summary}"""
        pos_neg_template = PromptTemplate(
            template=template, input_variables=["sec_summary"]
        )
        pos_neg_chain = LLMChain(
            llm=self.open_ai_llm, prompt=pos_neg_template, output_key="sec_final_output"
        )
        overall_chain = SequentialChain(
            input_variables=["stock", "summary"],
            chains=[sec_chain, pos_neg_chain],
            # Here we return multiple variables
            output_variables=["sec_final_output", "sec_summary"],
            verbose=True,
        )
        with get_openai_callback() as cb:
            statement = overall_chain(
                {"stock": self.ticker, "summary": self.stockllm.sec_analysis_agent()}
            )
            cb = {
                "Total Tokens": cb.total_tokens,
                "Prompt Tokens": cb.prompt_tokens,
                "Completion Tokens": cb.completion_tokens,
                "Total Cost (USD)": cb.total_cost,
            }
            statement["token_summary"] = cb
        return statement

    def input_from_user_zero_shot(self, query):
        ### Zero shot learning 
        template = """
"\n
{summary}\n"
Please predict sentiment classification of the above based on above text where sentiment can only be Strongly Positive, Positive, Strongly Negative, Negative, or Neutral. Only output the sentiment class, should be 1 or 2 words.:-
"""
        sec_template = PromptTemplate(template=template, input_variables=["summary"])
        return self.open_ai_llm(template.format(summary=query))
    
    def input_from_user_embedding_shot(self,query):
        classifications = ['Strongly Positive','Positive','Neutral','Negative','Strongly Negative']
        ### Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = open_ai_params["openai_api_key"])
        ## Create a faiss vector database
        faiss_classifications = FAISS.from_texts(classifications,embeddings)
        text = faiss_classifications.similarity_search_with_score(query,k = 1)[0][0].page_content
        return text
        
    def input_from_user_sentiment_file(self,file,type_of_file):
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

            entire_context = '\n\n'.join(entire_context)
            self.entire_context = entire_context
            
    def qachain_anthropic(self,vectorstore,query):
        if self.ticker == None:
            raise Exception("Ticker must be present")
        else:
            self.context_precursor()
            filter_dict = {'$and':[{'ticker':self.ticker},{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
            #documents = vectorstore.similarity_search(query,k = 1,filter = {'ticker':self.ticker})
        #documents = vectorstore.as_retriever(search_kwargs={"k": 1}).get_relevant_documents(query)
        k_count = max(len(set([doc.metadata['file_path'] for doc in documents])),3)*5
        if k_count != 5:
            documents = vectorstore.as_retriever(search_kwargs={"k": k_count,'filter':filter_dict}).get_relevant_documents(query)

        file_names = [doc.metadata['file_path'] for doc in documents]

        prompt = '''Use the following information to answer the question at the end in a coherent summary. 
    The below contains information about {ticker} and you are a financial analyst
    {context_precursor}
    Question: {question}
    Think step by step and be as detailed as possible. 
    Do not mention anything in your response about the context/information'''.format(ticker = self.ticker,context_precursor = self.entire_context,question = query )
        return prompt,file_names
    def qachain(self,vectorstore,query):
        if self.ticker == None:
            filter_dict = {'$and':[{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
        else:
            filter_dict = {'$and':[{'ticker':self.ticker},{'metadata':{'$ne':'Sentiment News'}}]}
            documents = vectorstore.as_retriever(search_kwargs={"k": 5,'filter':filter_dict}).get_relevant_documents(query)
            #documents = vectorstore.similarity_search(query,k = 1,filter = {'ticker':self.ticker})
        #documents = vectorstore.as_retriever(search_kwargs={"k": 1}).get_relevant_documents(query)
        k_count = max(len(set([doc.metadata['file_path'] for doc in documents])),3)*5
        if k_count != 5:
            documents = vectorstore.as_retriever(search_kwargs={"k": k_count,'filter':filter_dict}).get_relevant_documents(query)


    #page_content = vectorstore.as_retriever(search_kwargs={"k": 10}).get_relevant_documents(query)
        page_content = '\n\n'.join([doc.page_content for doc in documents])
        file_names = [doc.metadata['file_path'] for doc in documents]
        meta_data = documents[0].metadata
       # file_path = 
        context_precursor =  '''The below contains information about {} and you are a financial analyst'''.format(meta_data['ticker'])
       # import pdb;pdb.set_trace()
        prompt_template = """Use the following information to answer the question at the end in a coherent summary. 
    {context_precursor}
    {page_content}
    Question: {question}
    Think step by step.Do not mention "As per the information or context" in your response.
    """
        prompt = prompt_template.format(context_precursor = context_precursor,page_content = page_content,question = query)
        context_full_doc = []
        return prompt,documents
    
    def process_file_names(self,file_names):
        csv_filter = [file_name for file_name in file_names if '.csv' in file_name]
        df = pd.read_csv(csv_filter[0])
        df.rename(columns = {'Unnamed: 0':'Description'},inplace = True)
        return df       
