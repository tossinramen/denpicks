import pandas as pd
from ft5_model import df_full, model_ft5, feature_cols # Import from your main script

# 1. Champion Impact on FT5
# We look at Team Blue's champs (1-5)
blue_champs = ['Champ 1', 'Champ 2', 'Champ 3', 'Champ 4', 'Champ 5']
df_ft5 = df_full[df_full['FT5 Winner'] != 'N/A'].copy()

impact_list = []
unique_champs = pd.unique(df_ft5[blue_champs].values.ravel())

for champ in unique_champs:
    # Filter games where the champ was on Team Blue
    games = df_ft5[df_ft5[blue_champs].eq(champ).any(axis=1)]
    if len(games) > 10:  # Increased to 10 for your larger dataset
        rate = games['ft5_target'].mean()
        impact_list.append({'Champion': champ, 'FT5_Win_Rate': round(rate, 3), 'Sample': len(games)})

top_10 = pd.DataFrame(impact_list).sort_values('FT5_Win_Rate', ascending=False).head(10)

print("\n--- TOP 10 CHAMPIONS BY FT5 IMPACT (Global Data) ---")
print(top_10.to_string(index=False))

# 2. Feature Importance
importances = model_ft5.feature_importances_
print("\n--- FEATURE IMPORTANCE: What drives the Race to 5? ---")
for name, imp in zip(feature_cols, importances):
    print(f"{name}: {imp:.4f}")