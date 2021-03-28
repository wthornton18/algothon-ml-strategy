
import yfinance as yf
assettype = ['currency', 'bond', 'commodity, 'stock']
for asset in assettype:
    f = open(f'{asset}-symbols.txt', 'r')
    assets = []
    for line in f:
        assets.append(line.strip())
    f.close()
    data = yf.download(assets, start='2005-1-1', end='2020-12-30')
    data['Adj Close'].to_csv(f'{asset}-prices.csv')
    data['Volume'].to_csv(f'{asset}-volumes.csv')

