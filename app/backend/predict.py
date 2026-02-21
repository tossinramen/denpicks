import pandas as pd
import numpy as np
from ft5model import model_ft5, model_win, le, feature_cols

def get_prediction(team_blue, team_red, champs_blue, champs_red):
    """
    Input: 
    - team_blue, team_red: Strings (e.g., 'T1', 'Gen.G')
    - champs_blue, champs_red: Lists of 5 strings each
    """
    
    input_data = [team_blue, team_red] + champs_blue + champs_red
    match_df = pd.DataFrame([input_data], columns=feature_cols)

    
    try:
        encoded_match = match_df.copy()
        for col in feature_cols:
            encoded_match[col] = le.transform(match_df[col].astype(str))
    except ValueError as e:
        return f"Error: One of these teams or champions isn't in your 2025/2026 data: {e}"


    # [0] is the chance of Team Red winning, [1] is the chance of Team Blue winning
    ft5_prob = model_ft5.predict_proba(encoded_match)[0][1]
    win_prob = model_win.predict_proba(encoded_match)[0][1]

    print(f"\n--- PREDICTION FOR: {team_blue} vs {team_red} ---")
    print(f"Chance of {team_blue} getting FT5: {ft5_prob:.2%}")
    print(f"Chance of {team_blue} winning game: {win_prob:.2%}")

# Example Usage:
if __name__ == "__main__":
    # Replace these with any draft you want to test
    t_blue = "Cloud9"
    t_red = "FlyQuest"
    c_blue = ["Gnar", "Wukong", "Ryze", "Sion", "Senna"]
    c_red = ["Renekton", "Zaahen", "Taliyah", "Corki", "Nautilus"]
    
    get_prediction(t_blue, t_red, c_blue, c_red)