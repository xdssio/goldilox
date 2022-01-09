import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pd.DataFrame({'feature1': [1, 2, 3], 'new_feature': [2, 2, 3],
              'xgboost': [1, 1, 1], 'lightgbm': [1, 1, 2],
              'model1_explanation': [{'feature1': 1.5, 'new_feature': -1.1}, {'feature1': 1.5, 'new_feature': -1.5},
                                     {'feature1': 1.5, 'new_feature': 1.1}]})
