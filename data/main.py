import yaml
from datacreator import StockData,EconomyData
from openbb_terminal.sdk import openbb


# Open the YAML file and load its contents into a Python object
with open('apis.yaml', 'r') as file:
    yaml_data = yaml.load(file, Loader=yaml.FullLoader)

# Convert the YAML data to a dictionary
data_dict = dict(yaml_data)
openbb.keys.av(key = data_dict['OPENBB']['ALPHA_VANTAGE_KEY'], persist = True)
openbb.keys.fmp(key = data_dict['OPENBB']['FINANCIALMODELLING_AND_PREP_KEY'],persist = True)
openbb.keys.polygon(key = data_dict['OPENBB']['POLYGON_KEY'],persist = True)
openbb.keys.finnhub(key = data_dict['OPENBB']['FINNHUB_KEY'],persist = True)
openbb.keys.fred(key = data_dict['OPENBB']['FRED_KEY'],persist = True)