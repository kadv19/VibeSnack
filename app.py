import streamlit as st
import pandas as pd
import model_utils
from datetime import datetime

st.set_page_config(page_title="VibeSnack", page_icon="🍿", layout="wide")

# Load Model
@st.cache_resource
def get_model():
    return model_utils.load_model()

model = get_model()

# Sidebar
st.sidebar.title("🍿 VibeSnack")
st.sidebar.markdown("Your tiny, delightful snack recommender.")

if st.sidebar.button("Retrain Model"):
    with st.spinner("Training..."):
        import train_model
        train_model.train()
        st.cache_resource.clear()
        model = get_model()
    st.sidebar.success("Model retrained!")

# Main UI
st.title("What's the vibe? 🤔")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tell me about yourself")
    
    # Time
    now = datetime.now()
    current_hour = now.hour
    hour = st.number_input("Time (Hour 0-23)", min_value=0, max_value=23, value=current_hour)
    
    # Mood
    mood = st.selectbox("Mood", ["happy", "sad", "bored", "stressed", "energetic", "lazy"])
    
    # Hunger
    hunger = st.slider("Hunger Level", 1, 5, 3)
    
    # Diet
    diet = st.radio("Diet", ["veg", "non-veg"])
    
    # Context
    context = st.selectbox("Context", ["none", "studying", "gaming", "chilling", "gym"])
    
    if st.button("Recommend Snack 🚀", type="primary"):
        user_input = {
            "hour": hour,
            "mood": mood,
            "hunger": hunger,
            "diet": diet,
            "context": context
        }
        
        if model:
            predictions = model_utils.predict_snack(model, user_input, top_k=5)
            st.session_state['predictions'] = predictions
            st.session_state['user_input'] = user_input
            st.session_state['current_index'] = 0
        else:
            st.error("Model not loaded. Please train the model first.")

with col2:
    if 'predictions' in st.session_state:
        preds = st.session_state['predictions']
        idx = st.session_state.get('current_index', 0)
        
        if idx < len(preds):
            snack = preds[idx]
            
            st.subheader("I recommend...")
            st.markdown(f"## **{snack['name']}**")
            
            # Tags
            st.write(f"Tags: {', '.join(snack['tags'])}")
            
            # Message
            msg = model_utils.format_personalized_message(st.session_state['user_input'], snack['name'])
            st.info(msg)
            
            # Actions
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Accept ✅", key=f"accept_{idx}"):
                    model_utils.update_user_history(snack['id'])
                    st.toast("Saved to your history — used to personalize later!")
                    st.balloons()
            
            with c2:
                if st.button("Try another 🔄", key=f"next_{idx}"):
                    if idx + 1 < len(preds):
                        st.session_state['current_index'] = idx + 1
                        st.rerun()
                    else:
                        st.warning("No more recommendations!")
            
            # Why this snack?
            with st.expander("Why this snack?"):
                st.write(f"Model Probability: {snack['prob']:.2f}")
                st.write("Based on your inputs, this snack had the highest score matching your vibe.")
                
            # Alternatives (Static list of next 3)
            st.divider()
            st.caption("Alternatives:")
            alternatives = preds[idx+1:idx+4]
            if alternatives:
                for alt in alternatives:
                    st.write(f"- **{alt['name']}** ({alt['prob']:.2f})")
        else:
            st.write("No more recommendations. Try changing your inputs!")

