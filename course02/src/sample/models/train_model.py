# read data from csv file to pandas
df = pd.read_csv('data/iris.csv')
# split data into train and test
train, test = train_test_split(df, test_size=0.2) 