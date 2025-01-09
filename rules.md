Always activate the virtual environment before installing packages. 
Do not change my source code without permission.
Everytime you install a new package, add it to requirements.txt file.
Remember to update docker files and requirements.txt when necessary. 
Respect the Load prices from CSV file section in the mvo.py file. 
filePath = "F:\Data Science\CafeF.SolieuGD.Upto24092024\myport2.csv"
df = pd.read_csv(filePath)
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
df.set_index(["Date"], inplace=True)
prices = df.dropna()
returns = prices.pct_change().dropna()
