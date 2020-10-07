import sys  
sys.path.insert(0, '../py')
from graviti import *

import json
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

count = 0

csvfiles =  glob.glob('/home/garner1/Work/pipelines/nucleAI/data/features/BRCA/*/features_summary.csv')
for data in csvfiles:
    if count == 0:
        df = pd.read_csv(data)
        df.rename(columns={'Unnamed: 0':'stats'}, inplace=True)
        df_unpivoted = df.melt(id_vars=['stats'], var_name='feature', value_name='value')
    else:
        df = pd.read_csv(data)
        df.rename(columns={'Unnamed: 0':'stats'}, inplace=True)
        df_unpivoted = df_unpivoted.append(df.melt(id_vars=['stats'], var_name='feature', value_name='value'), ignore_index=True)
    count += 1
    print(count)
# Save file
filename = 'brca_unpivoted_summary.gz'
df_unpivoted.to_csv(filename)

# # Plot distros
# import seaborn as sns

# sns.set_theme(style="darkgrid")
# sns.displot(
#     df_unpivoted, x="value", col="feature", row="stats",
#     binwidth=3, height=3, facet_kws=dict(margin_titles=True),
# )
# plt.savefig('brca_unpivoted_summary.pdf')
