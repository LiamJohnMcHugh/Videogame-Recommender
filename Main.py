import pandas as pd
import numpy as np
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True)

file_path = 'Video_Games.csv'
result = pd.read_csv(file_path)
#print(result['Name'].values)

plottingDataGlobalSales = result['Global_Sales'].values
plottingDataCriticScores = result['Critic_Score'].values
plottingDataNames = result['Name'].values

indices_to_remove = [i for i in range(len(plottingDataGlobalSales)) if plottingDataCriticScores[i] != plottingDataCriticScores[i]]

plottingDataGlobalSales = [v for i, v in enumerate(plottingDataGlobalSales) if i not in indices_to_remove]
plottingDataCriticScores = [v for i, v in enumerate(plottingDataCriticScores) if i not in indices_to_remove]
plottingDataNames = [v for i, v in enumerate(plottingDataNames) if i not in indices_to_remove]

for j in range(len(plottingDataGlobalSales)):
    print(plottingDataNames[j], end=' ')
    print(plottingDataGlobalSales[j], end=' ')
    print(plottingDataCriticScores[j])

