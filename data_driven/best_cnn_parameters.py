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
        study = optuna.load_study(study_name=f'optimizing_parameters_{variable}_{state}_pareto',
                                  storage=f'sqlite:///data_driven/optimize/db/optimizing_parameters_state_{state}_pareto.db')
        best_trials = study.best_trials
        if len(best_trials) > 1:
            res = []
            for best_trial in best_trials:
                best_trial_values = best_trial.values
                if best_trial_values is not None:
                    if np.inf not in best_trial_values:
                           res.append(best_trial_values+[best_trial._trial_id])
            res = np.array(res)
            df = pd.DataFrame(res).sort_values(by=[0, 1])
            best_params = optuna.Trial(
                study=study, trial_id=int(df.iloc[0, -1])).params
        else:
            best_params = best_trials[0].params
        best_params['variable'] = variable
        best_params['state'] = state
        best_trials_df.append(best_params)
        print(best_params)
        print(best_trials[0].duration, best_trials[0]._trial_id)

pd.DataFrame(best_trials_df).to_csv(
    './data_driven/optimize/best_cnn_params.csv', index=False)
