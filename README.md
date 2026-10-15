# AI Spam Message Detection System

A complete machine-learning portfolio project built around the supplied `spam.csv` dataset. The app classifies pasted text as **spam** or **ham** using a TF-IDF text representation and a trained scikit-learn classifier, with a Streamlit interface and a model evaluation dashboard.

> **Dataset transparency:** The supplied CSV contains SMS messages (`Category`, `Message`), not a dedicated email corpus. This project is therefore accurately described as an SMS/spam-message detector. It can accept email text as an input demo, but the evaluation metrics are based on SMS data.

## Features
- Data cleaning and exact duplicate removal
- Stratified train/test split
- TF-IDF word and bigram features
- Comparison of Multinomial Naive Bayes and Logistic Regression
- Selection by spam-class F1-score
- Accuracy, spam precision, spam recall, spam F1, classification report and confusion matrix
- Streamlit interface for interactive predictions
- Saved model and evaluation metrics

## Project structure
```text
ai_spam_message_detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── spam.csv
├── spam_model.joblib          # generated after training
├── model_metrics.json         # generated after training
└── evaluation_report.txt      # generated after training
```

## Setup on Windows (VS Code)

1. Install Python (Python 3.10–3.12 recommended).
2. Extract this folder and open it in VS Code.
3. Open Terminal in VS Code and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Train and evaluate:

```powershell
python train_model.py
```

5. Launch the app:

```powershell
streamlit run app.py
```

The terminal will show a local URL, usually `http://localhost:8501`.

## Dataset
The included `data/spam.csv` contains `Category` and `Message` columns. Labels are `ham` and `spam`. The training script cleans rows, removes exact duplicate messages before splitting, and uses a fixed random seed for reproducibility.

## Evaluation
The script prints and saves actual metrics after training. Do not quote an accuracy value in a resume until you have run the script and checked `evaluation_report.txt`. For imbalanced data, examine spam precision, recall and F1—not accuracy alone.

## GitHub upload
Before uploading:
- Keep `data/spam.csv` only if the dataset's source/licence permits redistribution.
- Include `README.md` and `requirements.txt`.
- You may choose not to commit the generated model file if repository size is a concern; document how to train it.
- Never commit private emails, API keys, credentials, or personal data.

## Resume bullet (edit only after verifying your implementation)
"Built an end-to-end spam-message classification app using Python, TF-IDF and scikit-learn; compared Naive Bayes and Logistic Regression using a held-out test set and deployed interactive predictions with Streamlit."

## Limitations
- Training data is SMS-focused and may not generalize to email.
- The model may miss new spam patterns and may flag legitimate messages.
- It does not analyze email headers, attachments, URLs, sender reputation or inbox contents.
- Use as a portfolio demo, not as the sole security decision-maker.
