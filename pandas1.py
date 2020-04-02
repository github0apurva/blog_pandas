
# =============================================================================
# Download Data from UCI database, define variable names and replace ? data points as NaN
# =============================================================================
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
wget.download(data_url,'crx.csv')
crx_base = pd.read_csv("crx.csv", names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"] )
crx_base.replace('?', np.NaN, inplace = True)
