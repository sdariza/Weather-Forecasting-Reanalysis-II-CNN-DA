import warnings
import optuna
import pandas as pd
import numpy as np


warnings.filterwarnings("ignore")

states = [0, 6, 12, 18]
variables = ['air', 'vwnd', 'uwnd']

best_trials_df = []
for variable in ['air', 'vwnd', 'uwnd']:
    for state in [0, 6, 12, 18]:
        study = None
        dict_data = {}
        study = optuna.load_study(study_name=f'optimizing_parameters_{variable}_{state}_pareto',
                                  storage=f'sqlite:///data_driven/optimize/db/optimizing_parameters_state_{state}_pareto.db')
        df = study.trials_dataframe()
        df = df.sort_values(by=['duration', 'values_0', 'values_1'], ascending=[
                            False, True, True])
        print(df.iloc[0])
        dict_data['alpha'], dict_data['kz_h'], dict_data['kz_w'], dict_data['lr'] = df.iloc[0, 6:10].values
        dict_data['variable'], dict_data['state'] = variable, state
        best_trials_df.append(dict_data)

pd.DataFrame(best_trials_df).to_csv(
    './data_driven/optimize/best_cnn_params.csv', index=False)
