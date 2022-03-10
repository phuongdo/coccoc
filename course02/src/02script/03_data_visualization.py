import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=20,10

df = pd.read_csv('data/VNIndex.csv')
print(df.head())

df = df.rename(columns={"<DTYYYYMMDD>": "Date", "<Close>": "Close"})
df = df.astype({"Close": float})
df = df.astype({"Date": str})

df["Date"] = pd.to_datetime(df.Date, format="%Y%m%d")
print(df.dtypes)



# For our prediction project, we will just need “Date” and “Close” columns. Let’s get rid of the other columns then.
df = df[['Date', 'Close']]
print(df.head())

df.index = df['Date']
plt.plot(df["Close"],label='Close Price history')

plt.show()
