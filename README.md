SMS Dataset
     ↓
Data Cleaning
     ↓
EDA
     ↓
Text Preprocessing
     ↓
TFIDF Vectorization
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Pickle Model
     ↓
Streamlit Web App
     ↓
Deployment (Render)


app.py

Start
  │
  │
  ▼
Import Libraries
(streamlit, pickle, nltk, string)
  │
  ▼
Create Porter Stemmer
ps = PorterStemmer()
  │
  ▼
Define transform_text() Function
(Text preprocessing pipeline)
  │
  ▼
Load Saved Files
vectorizer.pkl  → TF-IDF
model.pkl       → Trained ML Model
  │
  ▼
Streamlit UI Loads
Title: "Email/SMS Spam Classifier"
  │
  ▼
User Enters Message
st.text_area()
  │
  ▼
User Clicks "Predict"
st.button()
  │
  ▼
Preprocess Text
transform_text(input_sms)
  │
  ▼
Vectorize Text
tfidf.transform()
(Text → Numerical Vector)
  │
  ▼
Model Prediction
model.predict()
  │
  ▼
Check Result
result == 1 ?
  │
 ┌───────────────┐
 │               │
 ▼               ▼
Spam        Not Spam
 │               │
 ▼               ▼
Display Result
st.header()
  │
  ▼
End
