import json
import model_utils
import pandas as pd

def run_demo():
    print("Loading model...")
    model = model_utils.load_model()
    if not model:
        return

    print("Loading demo inputs...")
    with open("demo_inputs.json", "r") as f:
        inputs = json.load(f)

    print("\n--- VibeSnack Demo Run ---\n")
    
    for i, user_input in enumerate(inputs):
        print(f"Input {i+1}: {user_input}")
        
        preds = model_utils.predict_snack(model, user_input, top_k=3)
        
        if preds:
            top_snack = preds[0]
            msg = model_utils.format_personalized_message(user_input, top_snack['name'])
            print(f"Recommendation: {top_snack['name']} (Prob: {top_snack['prob']:.2f})")
            print(f"Message: {msg}")
            print("Alternatives:", ", ".join([p['name'] for p in preds[1:]]))
        else:
            print("No recommendation found.")
        
        print("-" * 30)

if __name__ == "__main__":
    run_demo()
