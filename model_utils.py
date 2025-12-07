import joblib
import pandas as pd
import numpy as np
import os
import json
import random
from sklearn.base import BaseEstimator, TransformerMixin

def get_time_category(hour):
    if 7 <= hour <= 11: return "morning"
    if 12 <= hour <= 16: return "afternoon"
    if 17 <= hour <= 20: return "evening"
    return "night"

class TimeCategoryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is expected to be a DataFrame or 2D array with 'hour' in first column if we select it
        # But ColumnTransformer passes the selected column.
        # If we select 'hour', we get a 1D/2D array of hours.
        hours = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X[:, 0]
        cats = [get_time_category(h) for h in hours]
        return pd.DataFrame(cats, columns=['time_of_day_category'])

# Define Snack Catalog (Duplicate of data_generator for simplicity, or could import)
def load_snack_catalog():
    try:
        with open("snack_catalog.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading snack catalog: {e}")
        return []

SNACK_CATALOG = load_snack_catalog()

MODEL_PATH = "models/snack_model.joblib"
HISTORY_FILE = "user_history.json"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("Model not found. Please train it first.")
        return None

def prepare_input(user_input):
    """
    Converts user input dict to DataFrame expected by model.
    Input keys: hour, mood, hunger, diet, context
    """
    return pd.DataFrame([user_input])

def get_snack_by_id(snack_id):
    for s in SNACK_CATALOG:
        if s['id'] == snack_id:
            return s
    return None

def predict_snack(model, user_input, top_k=3):
    """
    Returns top_k snack IDs and their probabilities.
    Also adjusts based on user history.
    """
    df = prepare_input(user_input)
    
    # Get probabilities
    probs = model.predict_proba(df)[0]
    classes = model.classes_
    
    # Create a dict of snack_id -> prob
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    
    # Boost from history
    history = load_user_history()
    total_history = sum(history.values())
    if total_history > 0:
        for sid, count in history.items():
            sid = int(sid)
            if sid in prob_dict:
                # Small boost: 1% per accept, capped at 10%
                boost = min(0.1, (count / total_history) * 0.2) 
                prob_dict[sid] += boost
    
    # Sort
    sorted_snacks = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    top_k_snacks = []
    
    # Filter by diet
    user_diet = user_input.get('diet')
    
    for sid, prob in sorted_snacks:
        snack = get_snack_by_id(sid)
        if snack:
            # Strict filtering based on user request
            if user_diet == "veg" and "non-veg" in snack['tags']:
                continue
            if user_diet == "non-veg" and "veg" in snack['tags']:
                continue
                
            top_k_snacks.append({
                "id": sid,
                "name": snack['name'],
                "prob": prob,
                "tags": snack['tags']
            })
            
        if len(top_k_snacks) >= top_k:
            break
            
    return top_k_snacks

def format_personalized_message(user_input, snack_name):
    """
    Generates a friendly message.
    """
    hour = user_input.get('hour')
    mood = user_input.get('mood')
    hunger = user_input.get('hunger')
    diet = user_input.get('diet')
    context = user_input.get('context')
    
    time_str = f"{hour}:00" # Simplified
    
    msg = f"It's around {time_str} and you're feeling {mood}. "
    
    if hunger >= 4:
        msg += "You're pretty hungry! "
    elif hunger <= 2:
        msg += "Just looking for a light nibble? "
        
    if context and context != "none":
        msg += f"Since you're {context}, "
    
    msg += f"I recommend **{snack_name}**."
    
    return msg

def generate_explanation(user_input, snack):
    """
    Generates a detailed explanation for why a snack was chosen,
    focusing on the snack's attributes and how they match the user.
    """
    reasons = []
    
    tags = snack.get('tags', [])
    is_heavy = snack.get('heavy', False)
    name = snack.get('name', '')
    price = snack.get('price', 'medium')
    
    # User Context
    hour = user_input.get('hour')
    hunger = user_input.get('hunger')
    context = user_input.get('context')
    mood = user_input.get('mood')

    # 1. Name-based overrides (Specific flavor text)
    if "Yogurt" in name:
        reasons.append("Creamy and protein-packed.")
    elif "Banana" in name or "Fruit" in name or "Apple" in name:
        reasons.append("Nature's own fast food.")
    elif "Chocolate" in name:
        reasons.append("A classic mood booster.")
    elif "Coffee" in name:
        reasons.append("For that caffeine kick.")

    # 2. Hunger Matching
    if hunger >= 4:
        if is_heavy:
            reasons.append("Since you're very hungry, this substantial snack will fill you up.")
        elif "healthy" in tags:
            reasons.append("A high-volume, healthy option to satisfy your hunger.")
        else:
            reasons.append("A nice portion to help curb that major hunger.")
    elif hunger <= 2:
        if not is_heavy:
            reasons.append("It's light and won't ruin your appetite.")
        elif "sweet" in tags:
            reasons.append("A small sweet treat just for the taste.")
        else:
            reasons.append("A bit indulgent, but perfect if you want just one satisfying bite.")

    # 3. Context & Tag Combinations
    if context == "gaming":
        if "healthy" in tags:
             reasons.append("Fresh and clean - keeps your hands grease-free for gaming.")
        elif "quick" in tags and not is_heavy:
             reasons.append("Easy to pop in your mouth between rounds.")
        elif is_heavy:
             reasons.append("Hearty fuel for a long gaming session.")
        else:
             reasons.append("Good for a break between matches.")
             
    elif context == "studying":
        if "healthy" in tags:
            reasons.append("Brain food to keep you focused without the crash.")
        elif "sweet" in tags:
            reasons.append("A little sugar rush to keep you going.")
        elif "savory" in tags:
            reasons.append("A savory distraction to reward your hard work.")
            
    elif context == "gym":
        if "healthy" in tags:
            reasons.append("Great for fueling up or recovering.")
        elif is_heavy:
             reasons.append("Good for bulking up!")
        else:
            reasons.append("You earned a treat!")
            
    elif context == "chilling":
        if "savory" in tags:
            reasons.append("Perfect savory companion for relaxing.")
        elif "sweet" in tags:
            reasons.append("Sweet comfort food for downtime.")
        elif "healthy" in tags:
            reasons.append("A refreshing snack to chill with.")

    # 4. Time of Day
    if 7 <= hour <= 10:
        if "healthy" in tags:
            reasons.append("A healthy start to your morning.")
        elif "sweet" in tags:
            reasons.append("A sweet breakfast treat.")
        else:
            reasons.append("A tasty morning bite.")
            
    elif 14 <= hour <= 16:
        if "healthy" in tags:
            reasons.append("A refreshing afternoon pick-me-up.")
        elif "sweet" in tags:
            reasons.append("Perfect for that afternoon sugar craving.")
        elif "savory" in tags:
            reasons.append("A savory kick to wake you up.")
        else:
            reasons.append("Beats the afternoon slump.")
            
    elif 20 <= hour <= 23:
        if is_heavy:
            reasons.append("A hearty late-night meal.")
        elif "healthy" in tags:
             reasons.append("Light enough to not disrupt your sleep.")
        else:
            reasons.append("A light late-night munch.")

    # 5. Price Logic (Subtle)
    if price == "low" and not reasons:
        reasons.append("Great value for a quick bite.")
    elif price == "high" and mood == "sad":
        reasons.append("Treat yourself, you deserve it.")

    # 6. Fallback / Generic Tag mentions
    if not reasons:
        if "spicy" in tags:
            reasons.append("Spices things up a bit!")
        elif "sweet" in tags:
            reasons.append("Satisfies your sweet tooth.")
        elif "healthy" in tags:
            reasons.append("A guilt-free choice.")
        else:
            reasons.append("Matches your current vibe perfectly.")
            
    return " ".join(reasons)

def load_user_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def update_user_history(snack_id):
    history = load_user_history()
    sid_str = str(snack_id)
    history[sid_str] = history.get(sid_str, 0) + 1
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)
