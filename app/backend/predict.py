import pandas as pd
import numpy as np
from ft5model import model_ft5, model_win, le, feature_cols

def get_prediction(team_blue, team_red, champs_blue, champs_red):
    """
    Input: team strings and lists of 5 champ strings each
    """
    input_data = [team_blue, team_red] + champs_blue + champs_red
    match_df = pd.DataFrame([input_data], columns=feature_cols)
    
    try:
        encoded_match = match_df.copy()
        for col in feature_cols:
            encoded_match[col] = le.transform(match_df[col].astype(str))
    except ValueError as e:
        return f"Error: {e}. Check if you misspelled a champion or team name."

    # Probabilities for Team Blue (index 1)
    ft5_prob = model_ft5.predict_proba(encoded_match)[0][1]
    win_prob = model_win.predict_proba(encoded_match)[0][1]

    print(f"\n--- FULL DRAFT PREDICTION: {team_blue} vs {team_red} ---")
    print(f"Chance of {team_blue} getting FT5: {ft5_prob:.2%}")
    print(f"Chance of {team_blue} winning game: {win_prob:.2%}")

if __name__ == "__main__":
    # Test a specific draft
    get_prediction(
        team_blue="T1", 
        team_red="Gen.G", 
        champs_blue=["Rumble", "Nocturne", "Ahri", "Varus", "Nautilus"], 
        champs_red=["KSante", "Vi", "Taliyah", "Ashe", "Braum"]
    )