<details open="open">
<summary>Table of Contents</summary>

- [About](#about)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Cookiecutter template](#cookiecutter-template)
    - [Manual setup](#manual-setup)
    - [Variables reference](#variables-reference)
- [Roadmap](#roadmap)
- [Support](#support)
- [Acknowledgements](#acknowledgements)

</details>

---

# About

<table>
<tr>
<td>
The purpose of this repository is to enhance investment research and harness the capabilities of large language models (LLMs). Currently, there are two features available in this repository:

- Sentiment analysis of tickers on a daily basis using a multi shot approach learning approach
- Sentiment analysis of propreitary data.

These features leverage the power of LLMs to provide valuable insights and information to investors, enabling them to make informed decisions. By utilizing these tools, investors can gain a deeper understanding of market sentiment and potential investment opportunities.

Key features of **Findastic**:

- Sentiment Analysis of Propreitary data(Implemented)
- Sentiment Analysis of News of stock tickers(currently 10 stocks are implemented for the POC)
- Summaries of SEC filings of stocks(To be implemented)
- Summaries of Federal Reserve bank news(To be implemented)
- Ability to query fundamental analysis of a ticker (e.g. stock performance, key risks) (to be implemented)

These features are designed to provide users with valuable insights into the market, keep them up-to-date with the latest developments, and help them make informed decisions about their investments. Findastic is a comprehensive platform/data providor that aims to offer a range of tools and resources to help investors stay ahead of the curve.

# **Getting Started**

## **Prerequisites**

```sh
pip install -r requirements.txt
```

### Usage
In the next step, we need to set the following API keys namely:-
| API                        | Link                                                                               | 
| -------------------------- | --------------------                                                               | 
| Open AI API Key            | [Link](https://platform.openai.com/account/api-keys)                               |
| Cohere API Key             | [Link](https://dashboard.cohere.ai/)                                               |
| AI21 API Key               | [Link](https://studio.ai21.com/account)                                            |
| Alphavantage API Key       | [Link](https://www.alphavantage.co/support/#api-key)                               |
| FinnHubb API Key           | [Link](https://finnhub.io/dashboard)                                               |
| Polygon API Key            | [Link](https://polygon.io/dashboard)                                               |
| Fred API Key               | [Link](https://fred.stlouisfed.org/docs/api/api_key.html)                          |
| Financial Model & Prep     | [Link](https://site.financialmodelingprep.com/developer/docs/dashboard/)           |
| Google Search              | [Link](https://serpapi.com/dashboard)                                              |

**Note**: Other than Open AI, the free version of the API keys should suffice for general individual academic research.
Cohere AI currently has a trial API key plan which allows 5000 requests per month while  AI21 is free till July 31st 2023. We use Cohere/AI21 wherever we see a indistinguishable output between OpenAI and the alternative. However, our experimentation process shows that OpenAI's text DaVinci model performs the best especially in reasoning,classification and segmentation.

## **Data Collection**

#### **API creation and collation**
Once we have all the API keys set up, we create a yaml file in the /data folder as follows:-<br>
Command:
```sh
cd data
```
**apis.yaml config sample**
```sh 
OPENBB:
  ALPHA_VANTAGE_KEY: 
  FINANCIALMODELLING_AND_PREP_KEY: 
  FINNHUB_KEY: 
  POLYGON_KEY: 
  FRED_KEY: 
 
STOCKS:
  - 'AAPL'
  - 'TSLA'
  - 'V'
  - 'MSFT'
  - 'AMZN'
  - 'NVDA'
  - 'META'
  - 'GOOG'
  - 'BRK-B'
  - 'JNJ'
  
LLMS:
  COHERE_API_KEY: 
  OPENAI_API_KEY: 
  AI21_API_KEY: 
```
This example above is for 10 stocks listed above. Please keep your stock tickers that you want analyzed by adding to the STOCKS key in the yaml. 

We are using [OpenBB](https://github.com/OpenBB-finance/OpenBBTerminal) as our data vendor as it is an open-source investment research platform. We are collecting the following information about stocks.

- Financial Ratios
- Balance Sheet
- News
- Income Statement
- Cash Flows
- News 
- Sentiment Scores using the NLTK vader model
- 5 year estimates about the stock
- Fraud Ratios(M score,Z score & B score)

In order to collect data, run the following command.

```sh 
python main.py
```
This would create folders for each stock along with the necessary analysis. 

## **Methodology**

#### **Sentiment Analysis of news**

The algorithm we use is as follows:-
1. Collect news about stocks from OpenBB FinHubb API. 
2. Our initial analysis showed that a lot of news headlines were not related to the stock. 
3. We used Levenshtein distance on the news headlines(using both ticker and company name) and filtered the news related to the stocks(Suprisingly LLMs do not perform well on this.)
4. Next, we use a few shot learning approach where we manually labelled around 100 headlines of GOOG and TSLA into 5 categories i.e. Strongly Negative, Negative, Neutral, Positive & Strongly Positive. 
5. The 100 manually labelled labels are fed into a FAISS vector database.
6. When feeding a new stock's headlines, a similarity search is performed with the 100 examples present in the database and ~5 examples are fetched. 
7. The LLM is prompted as follows:-
```sh 
Generate a sentiment score of a headline
 The outputs can only be [Strongly Negative,Negative,Neutral,Positive,Strongly Positive]
 Input: Tesla Should Consider Buying Ads. It Could Help the Stock.
 Output: Negative
 Input: Tesla Inc. stock falls Tuesday, still outperforms market
 Output: Negative
 Input: Tesla: the problem is Musk‚Äôs multitasking, not his stake
 Output: Negative

 Input: GM earnings preview; Tesla raises 2023 capital expenditure forecast
 Output:
```
Each headline has a different set of training examples depending on its similarity with the trained examples. 

#### **SEC filings summary and analysis**

The algorithm we use is as follows:-
1. Collect SEC summary for the ticker. 
2. Generate a coherent bullet point summary of the SEC summary. Our analysis shows that the SEC summary obtained from FinnHubb is not in a clean consumable information for the end user. We use Cohere to generate the summaries for cost efficiency. 
3. Separate the summary into Positive and Negative News.

A sample output is as follows:-

```sh 
Positive News
1. iPad net sales increased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to higher net sales of iPad and iPad Air.
2. Services net sales increased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to higher net sales from cloud services, the App Store and music.
3. Services gross margin increased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to higher Services net sales.
4. The Company's effective tax rate for the first quarter of 2023 was lower compared to the same quarter in 2022 due primarily to a higher U. S. federal R & D credit, lower state income taxes and a lower effective tax rate on foreign earnings, largely offset by lower tax benefits from share-based compensation.
```
```sh
Negative News:
1. Europe net sales decreased during the first quarter of 2023 compared to the same quarter in 2022 due to the weakness in foreign currencies relative to the U. S. dollar, which contributed to lower net sales of iPhone and Mac.
2. Japan net sales decreased during the first quarter of 2023 compared to the same quarter in 2022 due to the weakness of the yen relative to the U. S. dollar, which contributed to lower net sales of Services and Mac.
3. Mac net sales decreased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to lower net sales of MacBook Pro.
4. Wearables, Home and Accessories net sales decreased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to lower net sales of AirPods, partially offset by higher net sales of Watch.
5. Products Gross Margin Products gross margin decreased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to the weakness in foreign currencies relative to the U. S. dollar and lower Products volume.
6. Services gross margin percentage decreased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to the weakness in foreign currencies relative to the U. S. dollar and higher Services costs, partially offset by improved leverage.
7. iPhone iPhone net sales decreased during the first quarter of 2023 compared to the same quarter in 2022 due primarily to lower net sales from the Company’s new iPhone models launched in the fourth quarter of 2022.
```

#### **Roadmap**
Here is the roadmap to the following questions:-

1. Summaries of SEC filings of stocks
The SEC filings summaries are yet to be exposed to the API.
2. Q&A answering about stock financial performance.
Create a vector database on Faiss where financial documents can be added with the appropriate index. 




## Support

Reach out to the maintainer at one of the following places:
- [GitHub discussions](https://github.com/maneelusf)


