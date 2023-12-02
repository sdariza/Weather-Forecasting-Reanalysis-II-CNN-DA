import warnings
import optuna
import numpy as np
import pandas as pd

variable = 'air'
state = 0
warnings.filterwarnings("ignore")

states = [0, 6, 12, 18]
variables = ['air', 'vwnd', 'uwnd']

best_trial_dfs = []


for state in states:
    for variable in variables:
        study = optuna.load_study(study_name=f'optimizing_parameters_{variable}_{state}',
                          storage=f'sqlite:///data_driven/optimize/db/optimizing_parameters_state_{state}.db')
        best_trials = study.best_trials
        if len(best_trials) > 1:
            res = []
            for i in study.trials:
                h = i.values
                if None != h:
                    if np.inf not in h:
                        res.append(h+[i.number])
            res = np.array(res)
            df = pd.DataFrame(res).sort_values(by=[0,1])
            best_params = optuna.Trial(study,int(df.iloc[0,-1])).params
        else:
            best_params = best_trials[0].params
        best_params['variable'] = variable
        best_params['state'] = state
        best_trial_dfs.append(best_params)

pd.DataFrame(best_trial_dfs).to_csv('./data_driven/optimize/best_cnn_params.csv', index=False)
