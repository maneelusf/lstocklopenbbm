import yaml
from langchain.llms import Cohere, OpenAI, AI21
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains import SequentialChain
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fuzzywuzzy import fuzz, process
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
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
ai21_params = {
    "model": "j2-grande-instruct",
    "numResults": 1,
    "temperature": 0,
    "topP": 1,
    "ai21_api_key": yaml_data["LLMS"]["AI21_API_KEY"],
    "maxTokens": 25,
}


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


#     def news_agent(self):
#         file_path = f'./ticker/


class LLM_analysis:
    def __init__(self, ticker, open_ai_params, cohere_params, ai21_params):
        ### Requires both Cohere and OpenAI APIs
        self.ticker = ticker
        self.open_ai_params = open_ai_params
        self.cohere_params = cohere_params
        self.ai21_params = ai21_params
        self.cohere_llm = Cohere(**self.cohere_params)
        self.open_ai_llm = OpenAI(**self.open_ai_params)
        self.ai21_llm = AI21(**self.ai21_params)
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

    def news_chain_analysis(self):
        try:
            train_data_set = pd.read_csv(
                "../train/news_train.csv", delimiter="\t"
            ).to_dict("records")
        except:
            raise Exception("The training file does not exist")
        try:
            test_data_set = pd.read_csv(
                "../train/news_train.csv", delimiter="\t"
            ).to_dict("records")
        except:
            raise Exception("The training file does not exist")
        ### This is the example prompt for getting the sentiment buckets.
        example_prompt = PromptTemplate(
            input_variables=["headline", "Sentiment"],
            template="Example Input: {headline}\nExample Output: {Sentiment}",
        )
        example_selector_initial = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            train_data_set,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            CohereEmbeddings(cohere_api_key=cohere_params["cohere_api_key"]),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=30,
        )
        example_selector_following = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            train_data_set,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            CohereEmbeddings(cohere_api_key=cohere_params["cohere_api_key"]),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=2,
        )
        try:
            test_data_set = pd.read_csv(f"./ticker/{self.ticker}/news/c_news.csv")[
                ["headline", "datetime"]
            ]
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
            suffix="Input: {Headline}\nOutput:",
            # What inputs your prompt will receive
            input_variables=["Headline"],
        )
        similar_prompt_following = FewShotPromptTemplate(
            # The object that will help select examples
            example_selector=example_selector_following,
            # Your prompt
            example_prompt=example_prompt,
            # Customizations that will be added to the top and bottom of your prompt
            prefix="""Generate a sentiment score of a headline\n The outputs can only be 
        [Strongly Negative,Negative,Neutral,Positive,Strongly Positive]""",
            suffix="Input: {Headline}\nOutput:",
            # What inputs your prompt will receive
            input_variables=["Headline"],
        )
        validation_dict = []
        for x in test_data_set.to_records("dict"):
            if x.index == 0:
                # llm = self.cohere_llm(**self.cohere_ai_params)
                pred = self.open_ai_llm(
                    similar_prompt_initial.format(Headline=x["headline"])
                )
                validation_dict.append(pred)
            else:
                # llm = self.cohere_llm(**self.cohere_ai_params)
                pred = self.open_ai_llm(
                    similar_prompt_following.format(Headline=x["headline"])
                )
                validation_dict.append(pred)
        test_data_set["sentiment_bucket"] = validation_dict
        z = test_data_set.copy()
        z["sentiment_bucket"] = [x.strip() for x in z["sentiment_bucket"]]
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
        return z

    def input_from_user(self, query):
        template = """
"\n
{summary}\n"
Please predict sentiment classification of the above based on above text where sentiment can only be Strongly Positive, Positive, Strongly Negative, Negative, or Neutral. Only output the sentiment class, should be 1 or 2 words.:-
"""
        sec_template = PromptTemplate(template=template, input_variables=["summary"])
        return self.open_ai_llm(template.format(summary=query))
    
