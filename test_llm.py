# test llm
import pandas as pd
from llm import llm_analysis, StockLLM
import yaml
import os
with open('./data/apis.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

open_ai_params = {'max_tokens':512,'openai_api_key' : yaml_data['LLMS']['OPENAI_API_KEY']}
cohere_params = {'model':'command-xlarge-nightly','max_tokens':2202,\
                 'cohere_api_key' : yaml_data['LLMS']['COHERE_API_KEY'],'temperature':0,\
                'k': 0}
ai21_params = {'model':"j2-grande-instruct",'numResults':1,'temperature':0,\
 'topP':1,'ai21_api_key':yaml_data['LLMS']['AI21_API_KEY'],"maxTokens":25}

def test_news_chain_analysis():
    # create synthetic data
    test_frame = pd.DataFrame(
        {"headline":["This stock is really really good",
                     "This stock is really terrible"],
         "datetime":["2022-01-01","2022-01-02"]}
    )
    if not os.path.exists("./ticker/TEST/news/"): os.mkdir("./ticker/TEST/news/")
    test_frame.to_csv("./ticker/TEST/news/c_news.csv")

    # Initiate
    test_analysis = llm_analysis('TEST',open_ai_params,cohere_params,ai21_params)
    result = test_analysis.news_chain_analysis()

    # check on dataframe
    assert list(result.columns) == ['ticker','headline','datetime','sentiment_bucket','sentiment_score']
    assert result['sentiment_bucket'][0] in ["Strongly Positive","Positive"]
    assert result['sentiment_bucket'][1] in ["Strongly Negative","Negative"]

    # remove data
    if os.path.exists("./ticker/TEST/news/c_news.csv"): 
        os.remove("./ticker/TEST/news/c_news.csv")