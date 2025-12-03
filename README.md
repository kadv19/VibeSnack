# 🍿 VibeSnack

VibeSnack is a tiny, delightful Machine Learning app that recommends snacks based on your context (time, mood, hunger, diet).

## Features
- **Personalized Recommendations**: Uses a Random Forest classifier trained on synthetic data to suggest the perfect snack.
- **Context Aware**: Considers time of day, mood, hunger level, diet preference, and activity.
- **Learning**: "Accepting" a recommendation saves it to your history, slightly boosting that snack's probability in the future.
- **Explanations**: Tells you *why* a snack was chosen with a friendly message.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Data & Train Model**
    The app will automatically train the model on first run, but you can do it manually:
    ```bash
    python data_generator.py
    python train_model.py
    ```

## Running the App

Start the Streamlit UI:
```bash
streamlit run app.py
```

## Demo Script

Run a CLI demo with pre-defined inputs:
```bash
python run_demo.py
```

## Project Structure
- `app.py`: Streamlit user interface.
- `data_generator.py`: Creates `snack_data.csv` (synthetic dataset).
- `train_model.py`: Trains the Random Forest model and saves it to `models/`.
- `model_utils.py`: Helper functions for prediction and history.
- `demo_inputs.json`: Sample inputs for testing.
