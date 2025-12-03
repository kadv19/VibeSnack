import joblib
import pandas as pd
import numpy as np
import os
import json
import random
from train_model import TimeCategoryEncoder # Need to import custom class for unpickling if not using 'main' scope

# Define Snack Catalog (Duplicate of data_generator for simplicity, or could import)
SNACK_CATALOG = [
    {"id": 1, "name": "Masala Popcorn", "tags": ["spicy", "veg", "quick"], "price": "low", "heavy": False},
    {"id": 2, "name": "Chocolate Bar", "tags": ["sweet", "veg", "quick"], "price": "low", "heavy": False},
    {"id": 3, "name": "Paneer Sandwich", "tags": ["savory", "veg", "moderate"], "price": "medium", "heavy": True},
    {"id": 4, "name": "Chicken Wrap", "tags": ["savory", "non-veg", "quick"], "price": "medium", "heavy": True},
    {"id": 5, "name": "Fruit Salad", "tags": ["healthy", "veg", "moderate"], "price": "medium", "heavy": False},
    {"id": 6, "name": "Protein Shake", "tags": ["healthy", "veg", "quick"], "price": "medium", "heavy": False},
    {"id": 7, "name": "Samosa", "tags": ["spicy", "veg", "quick"], "price": "low", "heavy": True},
    {"id": 8, "name": "Nachos with Salsa", "tags": ["spicy", "veg", "quick"], "price": "low", "heavy": False},
    {"id": 9, "name": "Almonds & Raisins", "tags": ["healthy", "veg", "quick"], "price": "low", "heavy": False},
    {"id": 10, "name": "Instant Noodles", "tags": ["savory", "veg", "quick"], "price": "low", "heavy": True},
    {"id": 11, "name": "Grilled Fish", "tags": ["savory", "non-veg", "moderate"], "price": "high", "heavy": True},
    {"id": 12, "name": "Ice Cream Cup", "tags": ["sweet", "veg", "quick"], "price": "medium", "heavy": False},
]

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
    for sid, prob in sorted_snacks[:top_k]:
        snack = get_snack_by_id(sid)
        if snack:
            top_k_snacks.append({
                "id": sid,
                "name": snack['name'],
                "prob": prob,
                "tags": snack['tags']
            })
            
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
