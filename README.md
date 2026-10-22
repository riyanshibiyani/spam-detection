# Email Spam Detector

A beginner-friendly Streamlit app that classifies messages as spam or genuine using TF-IDF features and a Multinomial Naive Bayes model.

## Features

- Check a message and view estimated spam and genuine scores.
- Browse and try the labeled sample messages.
- Review recent checks during the current session.
- View the model's test metrics and confusion matrix.
- Use a dark theme with custom fonts.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the app:

   ```powershell
   streamlit run spam_detection.py
   ```

The app will open at `http://localhost:8501`.

## Note on predictions

This is a mini-project demo trained on a small set of sample messages included in the app. Its predictions and test accuracy are not a guarantee of real-world performance and should not be used as the sole basis for deciding whether a message is safe.
