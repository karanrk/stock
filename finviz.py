from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
# from datetime import date
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="

tickers = [["MSFT","AAPL"], ["T","SNAP"], ["COIN", "ABBV"]]

def get_news(tickers):
    news = {}
    for ticker in tickers:
        print(f'Getting news for {ticker}')
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'karan'})
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news[ticker] = html.find(id='news-table')
    return news

def get_title(collection):
    bucket = []
    for ticker in collection:
        ticker_data = collection[ticker]
        rows = ticker_data.findAll('tr')
        for _, row in enumerate(rows):
            title = row.a.text
            date_info = row.td.text.split(' ')
            # assumes the title always consists of date(for articles published each day) in the first line 
            if len(date_info) == 1:
                time = date_info[0]
            else:
                time = date_info[1]
                date = date_info[0]

            # print(f'{ticker} {date_str} {time} {title}')
            bucket.append([ticker, date, time, title])
    return bucket

def score_and_visualize(data):
    df = pd.DataFrame(data, columns=['ticker', 'date', 'time', 'title'])
    vader = SentimentIntensityAnalyzer()
    # print(vader.polarity_scores('i hate Apple as it creates an ecosystem and has locked out many users.'))
    df['compound'] = df['title'].apply(lambda title: vader.polarity_scores(title)['compound'])
    df['date'] = pd.to_datetime(df.date).dt.date
    # plt.figure(figsize=(10, 8))
    mean_df = df.groupby(['ticker', 'date']).mean()
    mean_df = mean_df.unstack().xs('compound', axis='columns').transpose()
    mean_df.plot(kind='bar')
    plt.show()

for tk_group in tickers:
    news = get_news(tk_group)
    bucket = get_title(news)
    score_and_visualize(bucket)
