import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
DATA_25 = 'app/datasets/lol_data_2025.csv'
DATA_26 = 'app/datasets/processed_data_2026.csv'

df_25 = pd.read_csv(DATA_25)
df_26 = pd.read_csv(DATA_26)
df_full = pd.concat([df_25, df_26], ignore_index=True)

# 2. Weighting & Labeling
df_full['weight'] = df_full['Year'].apply(lambda x: 1.0 if x == 2026 else 0.5)
df_full['ft5_target'] = (df_full['FT5 Winner'] == df_full['Team Blue']).astype(int)
df_full['win_target'] = (df_full['Winner'] == df_full['Team Blue']).astype(int)

# 3. Features & Encoding
feature_cols = ['Team Blue', 'Team Red', 'Champ 1', 'Champ 2', 'Champ 3', 'Champ 4', 'Champ 5', 
                'Champ 6', 'Champ 7', 'Champ 8', 'Champ 9', 'Champ 10']

le = LabelEncoder()
all_categorical_data = pd.concat([df_full[col] for col in feature_cols])
le.fit(all_categorical_data.astype(str))

def encode_df(df):
    encoded = df[feature_cols].copy()
    for col in feature_cols:
        encoded[col] = le.transform(df[col].astype(str))
    return encoded

X_all = encode_df(df_full)

# 4. Train Models
# Use a mask to exclude LPL N/A rows for the FT5 specific model
ft5_mask = df_full['FT5 Winner'] != 'N/A'
X_ft5 = X_all[ft5_mask]
y_ft5 = df_full.loc[ft5_mask, 'ft5_target']
w_ft5 = df_full.loc[ft5_mask, 'weight']

model_ft5 = RandomForestClassifier(n_estimators=100, random_state=42)
model_ft5.fit(X_ft5, y_ft5, sample_weight=w_ft5)

model_win = RandomForestClassifier(n_estimators=100, random_state=42)
model_win.fit(X_all, df_full['win_target'], sample_weight=df_full['weight'])

print("--- Models successfully trained on full draft data ---")