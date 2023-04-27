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
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)
- [Acknowledgements](#acknowledgements)

</details>

---

## About

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

## Getting Started
Before we get started, we require the following 
### Prerequisites

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
Cohere AI currently has a trial API key plan which allows 5000 requests per month while  AI21 
### Data Collection

#### API creation and collation
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
#### Data Collection
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

### Methodology

Please follow these steps for manual setup:

1. [Download the precompiled template](https://github.com/dec0dOS/amazing-github-template/releases/download/latest/template.zip)
2. Replace all the [variables](#variables-reference) to your desired values
3. Initialize the repo in the precompiled template folder

    `or`

    Move the necessary files from precompiled template folder to your existing project directory. Don't forget the `.github` directory that may be hidden by default in your operating system

#### Variables reference

Please note that entered values are case-sensitive.
Default values are provided as an example to help you figure out what should be entered.

> On manual setup, you need to replace only values written in **uppercase**.



> NOTICE: to use GitHub Discussions, you have to [enable it first](https://docs.github.com/en/discussions/quickstart).

## Roadmap

See the [open issues](https://github.com/dec0dOS/amazing-github-template/issues) for a list of proposed features (and known issues).

- [Top Feature Requests](https://github.com/dec0dOS/amazing-github-template/issues?q=label%3Aenhancement+is%3Aopen+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Top Bugs](https://github.com/dec0dOS/amazing-github-template/issues?q=is%3Aissue+is%3Aopen+label%3Abug+sort%3Areactions-%2B1-desc) (Add your votes using the 👍 reaction)
- [Newest Bugs](https://github.com/dec0dOS/amazing-github-template/issues?q=is%3Aopen+is%3Aissue+label%3Abug)

## Contributing

First off, thanks for taking the time to contribute! Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.

Please try to create bug reports that are:

- _Reproducible._ Include steps to reproduce the problem.
- _Specific._ Include as much detail as possible: which version, what environment, etc.
- _Unique._ Do not duplicate existing opened issues.
- _Scoped to a Single Bug._ One bug per report.

Please adhere to this project's [code of conduct](docs/CODE_OF_CONDUCT.md).

You can use [markdownlint-cli](https://github.com/igorshubovych/markdownlint-cli) to check for common markdown style inconsistency.

## Support

Reach out to the maintainer at one of the following places:

- [GitHub discussions](https://github.com/dec0dOS/amazing-github-template/discussions)
- The email which is located [in GitHub profile](https://github.com/dec0dOS)


## Acknowledgements

Thanks for these awesome resources that were used during the development of the **Amazing GitHub template**:

- <https://github.com/cookiecutter/cookiecutter>
- <https://github.github.com/gfm/>
- <https://tom.preston-werner.com/2010/08/23/readme-driven-development.html>
- <https://changelog.com/posts/top-ten-reasons-why-i-wont-use-your-open-source-project>
- <https://thoughtbot.com/blog/how-to-write-a-great-readme>
- <https://www.makeareadme.com>
- <https://github.com/noffle/art-of-readme>
- <https://github.com/noffle/common-readme>
- <https://github.com/RichardLitt/standard-readme>
- <https://github.com/matiassingers/awesome-readme>
- <https://github.com/LappleApple/feedmereadmes>
- <https://github.com/othneildrew/Best-README-Template>
- <https://github.com/mhucka/readmine>
- <https://github.com/badges/shields>
- <https://github.com/cjolowicz/cookiecutter-hypermodern-python>
- <https://github.com/stevemao/github-issue-templates>
- <https://github.com/devspace/awesome-github-templates>
- <https://github.com/cezaraugusto/github-template-guidelines>
- <https://github.com/frenck?tab=repositories>
- <https://docs.github.com/en/discussions/quickstart>
- <https://docs.github.com/en/actions>

