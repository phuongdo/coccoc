import pandas as pd

df = pd.read_csv('data/vn_index.csv')
print(df.head())

df = df.rename(columns={"<DTYYYYMMDD>": "Date", "<Close>": "Close"})
df = df.astype({"Close": float})
df = df.astype({"Date": str})

df["Date"] = pd.to_datetime(df.Date, format="%Y%m%d")
print(df.dtypes)

# For our prediction project, we will just need “Date” and “Close” columns. Let’s get rid of the other columns then.
df = df[['Date', 'Close']]
print(df.head())