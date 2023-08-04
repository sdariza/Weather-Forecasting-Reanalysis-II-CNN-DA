import warnings
import optuna
import pandas as pd

warnings.filterwarnings("ignore")

states = [0, 6, 12, 18]
variables = ['air', 'vwnd', 'uwnd']

best_trial_dfs = []

for state in states:
    for variable in variables:
        study = optuna.load_study(study_name=f'optimizing_parameters_{variable}_{state}', storage=f'sqlite:///optimize/db/optimizing_parameters_state_{state}.db')
        best_trial_id = study.best_trial.number
        best_trial_value = study.best_trial.value
        trials_df = study.trials_dataframe()
        best_trial_df = trials_df[trials_df['number'] == best_trial_id]
        best_trial_df = best_trial_df[['value', 'params_alpha', 'params_kernel_size', 'params_learning_rate']]
        best_trial_df['params_kernel_size'] = best_trial_df['params_kernel_size'].apply(tuple)
        best_trial_df['state'] = state
        best_trial_df['variable'] = variable
        best_trial_dfs.append(best_trial_df)

final_df = pd.concat(best_trial_dfs, ignore_index=True)

final_df.to_csv('./optimize/best_cnn_params.csv', index=False)
