import requests
from bs4 import BeautifulSoup as bs


def get_stock_dict(topn=100):
    stock_dict = {"kospi": {}, "kosdaq": {}}
    for market in [0, 1]:
        for page in range(1, int(topn / 50 + 1)):
            html = requests.get(f"https://finance.naver.com/sise/sise_market_sum.nhn?sosok={market}&page={page}")
            soup = bs(html.text, "html.parser")
            for tr in soup.findAll("tr"):
                stock_info = tr.findAll('a', attrs={'class', 'tltle'})
                if stock_info is None or stock_info == []:
                    continue

                stock_code = stock_info[0]["href"].split("=")[1]
                stock_name = stock_info[0].contents[-1]

                if market == 0:
                    stock_dict["kospi"][stock_name] = stock_code
                else:
                    stock_dict["kosdaq"][stock_name] = stock_code

    return stock_dict
