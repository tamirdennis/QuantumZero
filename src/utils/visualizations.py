import plotly.express as px
import pandas as pd


def scatter_fidelity_as_function_of_T(statistics_path):
    df = pd.read_csv(statistics_path)

    df_T_groups = df.groupby('T', as_index=False)
    mean_fid_per_T = df_T_groups.mean()[['Fidelity', 'T']]
    std_fid_per_T = df_T_groups.std()['Fidelity'].fillna(0)
    mean_fid_per_T['std'] = std_fid_per_T
    fig = px.scatter(mean_fid_per_T, x='T', y='Fidelity', error_y='std')
    fig.show()


scatter_fidelity_as_function_of_T('/home/tamirdenis/PycharmProjects/QuantumZero/output/run_statistics_qubo_better.csv')