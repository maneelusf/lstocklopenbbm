import yaml
from datacreator import StockData, EconomyData,Sentiment_Generator,
from openbb_terminal.sdk import openbb
import time
import pandas as pd

# Open the YAML file and load its contents into a Python object
with open("apis.yaml", "r") as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

# Convert the YAML data to a dictionary
data_dict = dict(yaml_data)
openbb.keys.av(key=data_dict["OPENBB"]["ALPHA_VANTAGE_KEY"], persist=True)
openbb.keys.fmp(
    key=data_dict["OPENBB"]["FINANCIALMODELLING_AND_PREP_KEY"], persist=True
)
openbb.keys.polygon(key=data_dict["OPENBB"]["POLYGON_KEY"], persist=True)
openbb.keys.finnhub(key=data_dict["OPENBB"]["FINNHUB_KEY"], persist=True)
openbb.keys.fred(key=data_dict["OPENBB"]["FRED_KEY"], persist=True)
with open("apis.yaml", "r") as file:
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
    "model": "j2-jumbo-instruct",
    "numResults": 1,
    "temperature": 0,
    "topP": 1,
    "ai21_api_key": yaml_data["LLMS"]["AI21_API_KEY"],
    "maxTokens": 25,
}
sandp = pd.read_csv('S&P500.csv')
list_of_stocks = list(sandp['Ticker'])

for stock in list_of_stocks:
    stock_data = StockData(stock)
    stock_data.initialize_folders()
    stock_data.analysis_file()
    stock_data.ratio_file()
    stock_data.cash_file()
    stock_data.est_file()
    stock_data.fraud_file()
    stock_data.income_file()
    stock_data.balance_file()
    stock_data.news_file()
    time.sleep(1)

#### Generate sentiment scores for all news
embeddings = OpenAIEmbeddings(openai_api_key = yaml_data["LLMS"]["OPENAI_API_KEY"])
sentiment_generation = Sentiment_Generator(embeddings)
sentiment_generation.collect_news_files()
sentiment_generation.get_scores()
sentiment_generation.save_scores()


#### Few Shot Approach(implemented only for 10 stocks)