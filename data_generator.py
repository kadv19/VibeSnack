import pandas as pd
import numpy as np
import random

# 1. Define Snack Catalog
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

MOODS = ["happy", "sad", "bored", "stressed", "energetic", "lazy"]
CONTEXTS = ["studying", "gaming", "chilling", "gym", "none"]
DIETS = ["veg", "non-veg"]

def get_time_category(hour):
    if 7 <= hour <= 11: return "morning"
    if 12 <= hour <= 16: return "afternoon"
    if 17 <= hour <= 20: return "evening"
    return "night"

def generate_data(num_samples=1000):
    data = []
    
    for _ in range(num_samples):
        # Sample features
        hour = random.randint(7, 23)
        time_cat = get_time_category(hour)
        mood = random.choices(MOODS, weights=[0.2, 0.1, 0.2, 0.2, 0.15, 0.15])[0]
        
        # Hunger correlated with time slightly (lunch/dinner peaks)
        if 12 <= hour <= 14 or 19 <= hour <= 21:
            hunger = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]
        else:
            hunger = random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.15, 0.05])[0]
            
        diet = random.choices(DIETS, weights=[0.7, 0.3])[0]
        context = random.choices(CONTEXTS, weights=[0.2, 0.2, 0.3, 0.1, 0.2])[0]
        
        # Rule-based scoring
        scores = {item["id"]: 0 for item in SNACK_CATALOG}
        
        for item in SNACK_CATALOG:
            # Diet constraint
            if diet == "veg" and "non-veg" in item["tags"]:
                scores[item["id"]] -= 1000
            
            # Mood preferences
            if mood == "stressed" or mood == "sad":
                if "sweet" in item["tags"] or "spicy" in item["tags"]: scores[item["id"]] += 3
            if mood == "energetic" or mood == "happy":
                if "healthy" in item["tags"]: scores[item["id"]] += 2
            if mood == "bored":
                if "spicy" in item["tags"] or "savory" in item["tags"]: scores[item["id"]] += 2
            
            # Hunger preferences
            if hunger >= 4:
                if item["heavy"]: scores[item["id"]] += 4
            elif hunger <= 2:
                if not item["heavy"]: scores[item["id"]] += 3
                
            # Time preferences
            if time_cat == "morning":
                if "healthy" in item["tags"] or "sweet" in item["tags"]: scores[item["id"]] += 2
            if time_cat == "afternoon":
                if item["heavy"]: scores[item["id"]] += 2 # Lunch time
            if time_cat == "night":
                if item["heavy"]: scores[item["id"]] -= 2 # Avoid heavy late
                if "spicy" in item["tags"]: scores[item["id"]] += 1
                
            # Context preferences
            if context == "gym":
                if "healthy" in item["tags"]: scores[item["id"]] += 5
            if context == "studying":
                if "quick" in item["tags"] and not item["heavy"]: scores[item["id"]] += 3
            if context == "gaming":
                if "quick" in item["tags"]: scores[item["id"]] += 3
            if context == "chilling":
                if "spicy" in item["tags"] or "sweet" in item["tags"]: scores[item["id"]] += 2

        # Select snack
        # 10% random noise
        if random.random() < 0.1:
            # Filter by diet at least
            valid_snacks = [s for s in SNACK_CATALOG if not (diet == "veg" and "non-veg" in s["tags"])]
            chosen_snack = random.choice(valid_snacks)
        else:
            # Choose max score
            # Add tiny random noise to scores to break ties
            for sid in scores:
                scores[sid] += random.uniform(0, 0.5)
            
            best_id = max(scores, key=scores.get)
            chosen_snack = next(s for s in SNACK_CATALOG if s["id"] == best_id)
            
        data.append({
            "hour": hour,
            "time_of_day_category": time_cat,
            "mood": mood,
            "hunger": hunger,
            "diet": diet,
            "context": context,
            "snack_category_label": chosen_snack["name"], # Using name as label for readability
            "snack_id": chosen_snack["id"]
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_data(1000)
    df.to_csv("snack_data.csv", index=False)
    print(f"Generated {len(df)} rows of synthetic data to snack_data.csv")
