import pandas as pd
from ft5model import df_full, model_ft5, feature_cols 

# 1. Global Champion Impact Logic
blue_slots = ['Champ 1', 'Champ 2', 'Champ 3', 'Champ 4', 'Champ 5']
red_slots = ['Champ 6', 'Champ 7', 'Champ 8', 'Champ 9', 'Champ 10']
all_slots = blue_slots + red_slots

df_ft5 = df_full[df_full['FT5 Winner'] != 'N/A'].copy()
unique_champs = pd.unique(df_ft5[all_slots].values.ravel())

impact_list = []
for champ in unique_champs:
    # Games where champ was on Blue side
    blue_games = df_ft5[df_ft5[blue_slots].eq(champ).any(axis=1)]
    blue_wins = blue_games['ft5_target'].sum()
    
    # Games where champ was on Red side
    red_games = df_ft5[df_ft5[red_slots].eq(champ).any(axis=1)]
    red_wins = (red_games['ft5_target'] == 0).sum() 
    
    total = len(blue_games) + len(red_games)
    
    if total > 15: # Filter for better statistical significance
        global_rate = (blue_wins + red_wins) / total
        impact_list.append({'Champion': champ, 'FT5_Global_Rate': round(global_rate, 3), 'Sample': total})

top_10 = pd.DataFrame(impact_list).sort_values('FT5_Global_Rate', ascending=False).head(10)

print("\n--- TOP 10 CHAMPIONS BY GLOBAL FT5 IMPACT ---")
print(top_10.to_string(index=False))

# 2. Feature Importance
importances = model_ft5.feature_importances_
print("\n--- FEATURE IMPORTANCE: Draft vs. Team ---")
for name, imp in zip(feature_cols, importances):
    print(f"{name}: {imp:.4f}")