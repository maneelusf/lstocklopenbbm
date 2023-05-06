### Initializing all variables
from openbb_terminal.sdk import openbb
import os
import pandas as pd
from datetime import datetime as datetime
from dateutil.relativedelta import relativedelta
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import numpy as np
import yaml
from fuzzywuzzy import process
from __init__ import LOGGING_DIR
from logger import Logger
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import Cohere, OpenAI, AI21
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader,CSVLoader
from langchain.document_loaders.base import BaseLoader
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import logging

with open("apis.yaml", "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)
data_dict = dict(yaml_data)
stock_summary = pd.read_json("https://www.sec.gov/files/company_tickers.json").T
stock_summary = stock_summary[["title", "ticker"]]
stock_summary.columns = ["Company", "Ticker"]
openbb.keys.av(key=data_dict["OPENBB"]["ALPHA_VANTAGE_KEY"], persist=True)
openbb.keys.fmp(
    key=data_dict["OPENBB"]["FINANCIALMODELLING_AND_PREP_KEY"], persist=True
)
openbb.keys.polygon(key=data_dict["OPENBB"]["POLYGON_KEY"], persist=True)
openbb.keys.finnhub(key=data_dict["OPENBB"]["FINNHUB_KEY"], persist=True)
openbb.keys.fred(key=data_dict["OPENBB"]["FRED_KEY"], persist=True)
SERP_API_KEY = data_dict["GOOGLESEARCH"]

LOGGER = Logger(name='ExtractItems').get_logger()
LOGGER.info(f'Saving log to {os.path.join(LOGGING_DIR)}\n')

class StockData:
    def __init__(
        self,
        ticker,
        ratios=True,
        cash=True,
        balance=True,
        est=True,
        fraud=True,
        income=True,
        analysis=True,
        news=True,
        sec=True,
    ):
        self.ticker = ticker
        self.ratios = ratios
        self.cash = cash
        self.balance = balance
        self.est = est
        self.fraud = fraud
        self.income = income
        self.analysis = analysis
        self.news = news
        self.sec = sec

    def initialize_folders(self):
        path = os.listdir("../ticker")
        if self.ticker not in path:
            os.system("mkdir ../ticker/{}".format(self.ticker))
            for folder in ["fa", "news", "output"]:
                os.system("mkdir  ../ticker/{}/{}".format(self.ticker, folder))
                LOGGER.info(f'{self.ticker} for {folder} created')
                

    def create_file(self, file, file_folder, file_name, file_type):
        file_path = "../ticker/{}/{}/{}".format(
            self.ticker, file_folder, file_name + file_type
        )
        if file_type == ".csv":
            file.to_csv(file_path)
            LOGGER.info(f'{self.ticker} for {file_name} created')
        if file_type == ".txt":
            f = open(file_path, "w")
            f.write(file)
            LOGGER.info(f'{self.ticker} for {file_name} created')
            f.close()

    def ratio_file(self):
        ### Return all financial ratios of the selected stock
        if self.ratios:
            try:
                df = openbb.stocks.fa.ratios(self.ticker)
                df = df[df.index != "Period"]
                df['ticker'] = self.ticker
                # df.to_csv('../ticker/{}/{}/{}'.format(self.ticker,'fa','ratios.csv'))
                self.create_file(df, "fa", "ratios", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for ratios failure')
                pass

    def cash_file(self):
        ### Return cash flow of the selected stock
        if self.cash:
            try:
                df = openbb.stocks.fa.cash(self.ticker).fillna(0)
                df.index = [x + '(figures in $M)' for x in df.index]
                df = df/1000000
                df['ticker'] = self.ticker
                self.create_file(df, "fa", "cash", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for cash flow failure')
                pass

    def balance_file(self):
        ### Return balance sheet of the selected stock
        if self.balance:
            try:
                df = openbb.stocks.fa.balance(self.ticker).fillna(0)
                df.index = [x + '(figures in $M)' for x in df.index]
                df = df/1000000
                df.insert(0,'ticker',self.ticker)
                self.create_file(df, "fa", "balance", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for balance failure')
                pass

    def est_file(self):
        ### Return estimates from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.est(self.ticker)[0].fillna(0)
                add_dict = {'Revenue':'(In $M)','EBIT':'(In $M)','EBITDA':'(In $M)','Net Profit':'(In $M)','Net Profit Adjusted':'(In $M)',
'Pre-Tax Profit':'(In $M)','Net Profit (Adjusted)':'(In $M)','EPS (Non-GAAP) ex. SOE':'(In $)',\
'EPS (GAAP)':'(In $)','Gross Income':'(In $M)','Cash Flow from Investing':'(In $M)',\
'Cash Flow from Financing':'(In $M)','Cash Flow from Operations':'(In $M)','Cash Flow per Share':'(In $)',\
'Free Cash Flow':'(In $M)','Free Cash Flow per Share':'(In $)','Book Value per Share':'(In $)',\
'Net Debt':'(In $M)','Research & Development Exp.':'(In $M)','Capital Expenditure':'(In $M)',\
'Selling, General & Admin. Exp.':'(In $M)','Shareholder’s Equity':'(In $M)','Total Assets':'(In $M)'}
                df.index = [x + add_dict[x] if x in add_dict.keys() else x for x in df.index]
                df.insert(0,'ticker',self.ticker)
                self.create_file(df, "fa", "est", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for estimates failure')
                pass

    def fraud_file(self):
        ### Return fraud ratios from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.fraud(self.ticker).fillna(0)
                if df.shape[0] == 3:
                    df.index = ['Manipulation score(Fraud)','Altman Z-score(Fraud)','McKee ratio(Fraud)']
                df = df[df.columns[::-1]]
                df.insert(0,'ticker',self.ticker)
                self.create_file(df, "fa", "fraud", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for fraud ratios failure')
                pass

    def income_file(self):
        ### Return income from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.income(self.ticker)
                df.index = [x + '(figures in $M)' for x in df.index]
                df = df/1000000
                df.insert(0,'ticker',self.ticker)
                self.create_file(df, "fa", "income", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for income statement failure')
                pass

    def analysis_file(self):
        ### Return analysis from analysts for the selected stock
        if self.analysis:
            try:
                if self.ticker == "META":
                    df = openbb.stocks.fa.analysis("FB")
                    df.insert(0,'ticker',self.ticker)
                else:
                    df = openbb.stocks.fa.analysis(self.ticker)
                    df.insert(0,'ticker',self.ticker)
                text = "".join(list(df["Sentence"]))
                self.create_file(text, "fa", "analysis_sec", ".txt")
            except:
                LOGGER.info(f'{self.ticker} for analysis failure')
                pass

    def news_file(self):
        ### Return analysis from analysts for the selected stock
        if self.news:
            try:
                ### News from FinnHub
                end_date = datetime.today().strftime("%Y-%m-%d")
                start_date = (datetime.today() - relativedelta(months=2)).strftime(
                    "%Y-%m-%d"
                )
                if self.ticker == "BRK-B":
                    df = openbb.stocks.ba.cnews(
                        "BRK.A", start_date=start_date, end_date=end_date
                    )
                    df.insert(0,'ticker',self.ticker)
                else:
                    df = openbb.stocks.ba.cnews(
                        self.ticker, start_date=start_date, end_date=end_date
                    )
                df = pd.DataFrame(df)[["related", "datetime", "headline", "summary"]]
                df["datetime"] = df["datetime"].apply(
                    lambda x: datetime.fromtimestamp(x)
                )
                choices = list(
                    stock_summary[stock_summary["Ticker"] == self.ticker].values[0]
                )
                result = pd.DataFrame(
                    [
                        process.extract(headline, choices, limit=2)
                        for headline in df["headline"]
                    ]
                )
                result.columns = choices
                result[choices[0]] = [x[1] for x in result[choices[0]]]
                result[choices[1]] = [x[1] for x in result[choices[1]]]
                result["headline"] = df["headline"]
                result["final_score"] = [
                    max(x, y) for x, y in zip(result[choices[0]], result[choices[1]])
                ]
                result = result[result["final_score"] > 50][["headline", "final_score"]]
                result["datetime"] = df["datetime"]
                # import pdb;pdb.set_trace()
                self.create_file(result, "news", "c_news", ".csv")
                ### Sentiment analysis of news
                if self.ticker == "BRK-B":
                    df = openbb.stocks.ba.snews("BRK.A")
                else:
                    df = openbb.stocks.ba.snews(self.ticker)
                df.columns = ["sentiment"]
                df = df[df["sentiment"] != 0]
                df.insert(0,'ticker',self.ticker)
                self.create_file(df, "news", "s_news", ".csv")
            except:
                LOGGER.info(f'{self.ticker} for news failure')
                pass


class EconomyData:
    def __init__(self, fed_news=True, treasury_data=True):
        self.fed_news = fed_news
        self.treasury_news = treasury_data

    def fed_file(self):
        if self.fed_news:
            search = GoogleSearch(
                {
                    "q": "US Federal Reserve Bank news",
                    "api_key": SERP_API_KEY,
                    "tbm": "nws",
                }
            )
            result = search.get_dict()
            sources = ["cnbc", "reuters", "forbes"]
            results = [
                x["link"]
                for x in result["news_results"]
                if ("Fed" in x["title"]) and (x["source"].lower() in sources)
            ]
            link_text = [self.text_link(link) for link in results]
            link_df = pd.DataFrame(link_text, columns=["article"])
            self.create_file(link_df, "fed_news", ".csv")

    def treasury_file(self):
        if self.treasury_news:
            try:
                result = openbb.economy.macro(
                    parameters=["CPI", "M3YD", "Y10YD", "URATE"]
                )[0]
                self.create_file(result, "economy_indicators", ".csv")
            except:
                pass

    def text_link(self, link):
        try:
            r = requests.get(link)
            soup = BeautifulSoup(r.content, "html.parser")
            text = ""
            for a in soup.find_all("p"):
                text = text + a.text
        except:
            text = ""
        return text

    def create_file(self, file, file_name, file_type):
        file_path = "../economy/{}".format(file_name + file_type)
        if file_type == ".csv":
            file.to_csv(file_path)
        if file_type == ".txt":
            f = open(file_path, "w")
            f.write(file)
            f.close()
            
class FewShot_Sentiment_Generator:
    def __init__(self,ticker,open_ai_params,cohere_params,ai21_params):
        self.ticker = ticker
        self.open_ai_params = open_ai_params
        self.cohere_params = cohere_params
        self.ai21_params = ai21_params
        self.cohere_llm = Cohere(**self.cohere_params)
        self.open_ai_llm = OpenAI(**self.open_ai_params)
        self.ai21_llm = AI21(**self.ai21_params)
        self.stockllm = StockLLM(self.ticker)
 
    def news_chain_analysis(self):
            try:
                train_data_set = pd.read_csv(
                    "../train/news_train.csv", delimiter=","
                )[['headline','sentiment']].to_dict('records')
            except:
                raise Exception("The training file does not exist")
            ### This is the example prompt for getting the sentiment buckets.
            example_prompt = PromptTemplate(
                input_variables=["headline", "sentiment"],
                template="Example Input: {headline}\nExample Output: {sentiment}",
            )
            example_selector_initial = SemanticSimilarityExampleSelector_from_db.from_examples(
                # This is the list of examples available to select from.
                train_data_set,
                # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                OpenAIEmbeddings(openai_api_key=self.open_ai_params["openai_api_key"]),
                # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                FAISS,
                # This is the number of examples to produce.
                k=30,
            )

            example_selector_following = SemanticSimilarityExampleSelector_from_db.from_examples(
                # This is the list of examples available to select from.
                train_data_set,
                # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                OpenAIEmbeddings(openai_api_key=self.open_ai_params["openai_api_key"]),
                # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                FAISS,
                # This is the number of examples to produce.
                k=2,
            )
            try:
                test_data_set = pd.read_csv(f"../ticker/{self.ticker}/news/c_news.csv")[
                    ["headline", "datetime"]
                ]
                test_data_set.insert(0,'ticker',self.ticker)
                test_data_set.dropna(inplace = True)
            except:
                raise Exception(f"The news file of {self.ticker} does not exist")
            similar_prompt_initial = FewShotPromptTemplate(
                # The object that will help select examples
                example_selector=example_selector_initial,
                # Your prompt
                example_prompt=example_prompt,
                # Customizations that will be added to the top and bottom of your prompt
                prefix="""Generate a sentiment score of a headline\n The outputs can only be 
            [Strongly Negative,Negative,Neutral,Positive,Strongly Positive]""",
                suffix="Input: {headline}\nOutput:",
                # What inputs your prompt will receive
                input_variables=["headline"],
            )
            similar_prompt_following = FewShotPromptTemplate(
                # The object that will help select examples
                example_selector=example_selector_following,
                # Your prompt
                example_prompt=example_prompt,
                # Customizations that will be added to the top and bottom of your prompt
                prefix="""Generate a sentiment score of a headline\n The outputs can only be 
            [Strongly Negative,Negative,Neutral,Positive,Strongly Positive]""",
                suffix="Input: {headline}\nOutput:",
                # What inputs your prompt will receive
                input_variables=["headline"],
            )
            validation_dict = []
            for index,dict_ in enumerate(test_data_set.to_dict(orient='records')):
                #import pdb;pdb.set_trace()
                if index == 0:
                    # llm = self.cohere_llm(**self.cohere_ai_params)
                    pred = self.ai21_llm(
                        similar_prompt_initial.format(headline=dict_["headline"])
                    )
                    validation_dict.append(pred)
                else:
                    # llm = self.cohere_llm(**self.cohere_ai_params)
                    pred = self.ai21_llm(
                        similar_prompt_following.format(headline=dict_["headline"])
                    )
                    validation_dict.append(pred)
            test_data_set["sentiment_bucket"] = validation_dict
            z = test_data_set.copy()
            z["sentiment_bucket"] = [x.strip() for x in z["sentiment_bucket"]]
           # import pdb;pdb.set_trace()
            #z['sentiment_bucket'] = [return_sentiment_bucket(x) for x in z['sentiment_bucket']]
           # import pdb;pdb.set_trace()
            df = pd.DataFrame(
                {
                    "Strongly Positive": [0.8, 1],
                    "Positive": [0.6, 0.8],
                    "Neutral": [0.5, 0.51],
                    "Negative": [0.2, 0.4],
                    "Strongly Negative": [0, 0.2],
                }
            ).T.reset_index()
            df["index"] = df["index"].astype(str)
            df.columns = ["index", "lower_bound", "higher_bound"]
            z = z.merge(df, left_on="sentiment_bucket", right_on="index", how="outer")

            z["sentiment_score"] = [
                np.random.randint(x * 100, y * 100) / 100
                for x, y in zip(z["lower_bound"], z["higher_bound"])
            ]
            z["ticker"] = self.ticker
            final_cols = [
                "ticker",
                "headline",
                "datetime",
                "sentiment_bucket",
                "sentiment_score",
            ]
            z = z[final_cols]
            z = z.dropna()
            faiss = FAISS.load_local('faiss_sentiment',OpenAIEmbeddings(openai_api_key=self.open_ai_params["openai_api_key"]))
            df_insert = z[['headline','sentiment_bucket']]
            df_insert.columns = ['headline','sentiment']
           # import pdb;pdb.set_trace()
            df_insert.to_csv('sample.csv',index = False)
            loader = CSVLoader('sample.csv',source_column = 'headline')
            docs = loader.load()
           # import pdb;pdb.set_trace()
            faiss.add_documents(docs)
            os.remove('sample.csv')
            z.to_csv(f'../ticker/{self.ticker}/output/news_sentiment.csv')
            faiss.save_local(folder_path = 'faiss_sentiment')
            return z
    
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
    
class SemanticSimilarityExampleSelector_from_db(SemanticSimilarityExampleSelector):

    @classmethod
    def from_examples(
        cls,
        examples,
        embeddings,
        vectorstore_cls,
        k = 4,
        input_keys = None,
        db_path = 'faiss_sentiment',
        **vectorstore_cls_kwargs):
        """Create k-shot example selector using example list and embeddings.
        Reshuffles examples dynamically based on query similarity.
        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store
        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]

        else:
            #import pdb;pdb.set_trace()
            string_examples = ["\n".join(sorted_values(eg)) for eg in examples]
        try:
            vectorstore = vectorstore_cls.load_local(db_path)
        except:
            vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs)
            vectorstore.save_local(db_path)
        return cls(vectorstore=vectorstore, k=k, input_keys=input_keys)

def sorted_values(values):
    """Return a list of values in dict sorted by key."""
    #import pdb;pdb.set_trace()
    return ['{}:{}'.format(key,value) for key,value in values.items()]

class Sentiment_Generator:
    def __init__(self,embeddings):
        self.embeddings = embeddings
    
    def collect_news_files(self):
        main_file = []
        for stock in os.listdir('../ticker'):
            try:
                df = pd.read_csv(f'../ticker/{stock}/news/c_news.csv')
                df.insert(0,'ticker',stock)
                df = df[['ticker','headline','datetime']]
            except:
                pass
            main_file.append(df)
        self.document_file = pd.concat(main_file)
    
    def get_scores(self):
        docs = list(self.document_file['headline'])
        classifications = ['Strongly Positive','Positive','Neutral','Negative','Strongly Negative']
        ## Create a faiss vector database
        #import pdb;pdb.set_trace()
        faiss_classifications = FAISS.from_texts(classifications,self.embeddings)
        self.similarity_scores = []
        for x in docs:
            #import pdb;pdb.set_trace()
            self.similarity_scores.append(faiss_classifications.similarity_search_with_score(x,k = 1)[0][0].page_content)
        self.document_file['similarity'] = self.similarity_scores
        
    def save_scores(self):
        self.document_file.to_csv('output/sentiment_scores.csv')

        
### Own csvloader
class CSVLoader_v1(BaseLoader):
    """Loads a CSV file into a list of documents.
    Each document represents one row of the CSV file. Every row is converted into a
    key/value pair and outputted to a new line in the document's page_content.
    The source for each document loaded from csv is set to the value of the
    `file_path` argument for all doucments by default.
    You can override this by setting the `source_column` argument to the
    name of a column in the CSV file.
    The source of each document will then be set to the value of the column
    with the name specified in `source_column`.
    Output Example:
        .. code-block:: txt
            column1: value1
            column2: value2
            column3: value3
    """

    def __init__(
        self,
        file_path: str,
        source_column= None,
        encoding = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
    def load(self):
        """Load data into document objects."""

        docs = []
        with open(self.file_path,'r') as f:
            #import pdb;pdb.set_trace()
            ticker = self.file_path.split('/')[2]
            meta_data = self.file_path.split('.')[-2].split('/')[-1]
            dict1 = {'balance':'Balance Sheet','cash':'Cash Flow','income':'Income Statement',\
                    'ratios':'Key Financial Ratios','est':'Analyst Estimates','fraud':'Fraud Ratios',
                    'c_news':'News','s_news':'Sentiment News'}
            if meta_data in dict1.keys():
                meta_data = dict1[meta_data]
            metadata = {"ticker": ticker, "metadata": meta_data,"file_path": self.file_path}
            file_content = f.read()
        doc = Document(page_content=file_content, metadata=metadata)
        return [doc]

### Own textloader
class TextLoader_v1(BaseLoader):
    def __init__(
        self,
        file_path: str,
        source_column= None,
        encoding = None,
    ):
        self.file_path = file_path
        self.source_column = source_column
        self.encoding = encoding
    def load(self):
        """Load data into document objects."""

        docs = []
        with open(self.file_path,'r') as f:
            ticker = self.file_path.split('/')[2]
            meta_data = self.file_path.split('.')[-2].split('/')[-1]
            dict1 = {'analysis_sec':'SEC analysis summary'}
            if meta_data in dict1.keys():
                meta_data = dict1[meta_data]
            metadata = {"ticker": ticker, "metadata": meta_data,"file_path": self.file_path}
            file_content = f.read()
        doc = Document(page_content=file_content, metadata=metadata)
        return [doc]