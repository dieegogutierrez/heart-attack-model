import pickle
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# parameters

n_estimators=10
max_depth=6
output_file_path = '../models/model.bin'

# # Dataset

df = pd.read_csv('../data/heart.csv')

# # Columns with nulls/NA thall = 0 and caa = 4

df = df[(df.thall != 0) & (df.caa != 4)]

# # Split the data

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_train_full = df_train_full.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train_full = df_train_full['output'].values
y_test = df_test['output'].values

del df_train_full['output']
del df_test['output']

X_train_full = df_train_full.values
X_test = df_test.values

# # Final training with chosen model

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)

rf.fit(X_train_full, y_train_full)

y_pred = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print(f'auc={round(auc, 3)}')

# Save the model

with open(output_file_path, 'wb') as f_out:
    pickle.dump((rf), f_out)

print(f'the model is saved to {output_file_path}')