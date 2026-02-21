import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load and Weight Data
df_25 = pd.read_csv('lol_data_2025.csv')
df_26 = pd.read_csv('processed_data_2026.csv')

df_full = pd.concat([df_25, df_26])
df_full['weight'] = df_full['year'].apply(lambda x: 1.0 if x == 2026 else 0.5)

df_full['ft5_target'] = (df_full['ft5_winner'] == df_full['team1']).astype(int)
df_full['win_target'] = (df_full['winner'] == df_full['team1']).astype(int)


le = LabelEncoder()


# 4. Train the FT5 Model (Exclude LPL 'N/A' rows)
df_ft5 = df_full[df_full['ft5_winner'] != 'N/A']
model_ft5 = RandomForestClassifier(n_estimators=100)
model_ft5.fit(X_ft5, df_ft5['ft5_target'], sample_weight=df_ft5['weight'])

# 5. Train the Match Win Model (Include LPL)
model_win = RandomForestClassifier(n_estimators=100)
model_win.fit(X_win, df_full['win_target'], sample_weight=df_full['weight'])

print("Models trained. Use model_ft5.predict_proba() for percentage chances.")