### Initializing all variables
from openbb_terminal.sdk import openbb
import os
import pandas as pd
from datetime import datetime as datetime
from dateutil.relativedelta import relativedelta
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import yaml
from fuzzywuzzy import process

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

    def create_file(self, file, file_folder, file_name, file_type):
        file_path = "../ticker/{}/{}/{}".format(
            self.ticker, file_folder, file_name + file_type
        )
        if file_type == ".csv":
            file.to_csv(file_path)
        if file_type == ".txt":
            f = open(file_path, "w")
            f.write(file)
            f.close()

    def ratio_file(self):
        ### Return all financial ratios of the selected stock
        if self.ratios:
            try:

                df = openbb.stocks.fa.ratios(self.ticker)
                df = df[df.index != "Period"]
                # df.to_csv('../ticker/{}/{}/{}'.format(self.ticker,'fa','ratios.csv'))
                self.create_file(df, "fa", "ratios", ".csv")
            except:
                pass

    def cash_file(self):
        ### Return cash flow of the selected stock
        if self.cash:
            try:
                df = openbb.stocks.fa.cash(self.ticker).fillna(0)
                self.create_file(df, "fa", "cash", ".csv")
            except:
                pass

    def balance_file(self):
        ### Return balance sheet of the selected stock
        if self.balance:
            try:
                df = openbb.stocks.fa.balance(self.ticker).fillna(0)
                self.create_file(df, "fa", "balance", ".csv")
            except:
                pass

    def est_file(self):
        ### Return estimates from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.est(self.ticker)[0].fillna(0)
                self.create_file(df, "fa", "est", ".csv")
            except:
                pass

    def fraud_file(self):
        ### Return fraud ratios from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.fraud(self.ticker).fillna(0)
                df = df[df.columns[::-1]]
                self.create_file(df, "fa", "fraud", ".csv")
            except:
                pass

    def income_file(self):
        ### Return income from analysts for the selected stock
        if self.est:
            try:
                df = openbb.stocks.fa.income(self.ticker)
                self.create_file(df, "fa", "income", ".csv")
            except:
                pass

    def analysis_file(self):
        ### Return analysis from analysts for the selected stock
        if self.analysis:
            try:
                df = openbb.stocks.fa.analysis(self.ticker)
                text = "".join(list(df["Sentence"]))
                self.create_file(text, "fa", "analysis_sec", ".txt")
            except:
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
                self.create_file(df, "news", "s_news", ".csv")
            except:
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
            link_df = pd.DataFrame(y, columns=["article"])
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
