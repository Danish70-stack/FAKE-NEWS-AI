import os
import sys
import joblib
import speech_recognition as sr
import pyttsx3

# import preprocess
sys.path.append(os.path.dirname(__file__))
from preprocess import clean_text

# paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# load model
model = joblib.load(os.path.join(MODEL_DIR, "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# text to speech (initialize ONCE)
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text):
    """Safe text-to-speech"""
    engine.stop()  # prevent loop overlap
    engine.say(text)
    engine.runAndWait()

def listen_news():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak the news now...")
        speak("Please speak the news")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("📝 You said:", text)
        return text
    except sr.UnknownValueError:
        speak("Sorry, I could not understand.")
        return None
    except sr.RequestError:
        speak("Speech service is unavailable.")
        return None

def predict_text(text):
    """Pure ML prediction (NO voice here)"""
    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    proba = model.predict_proba(vec)[0]
    label = model.predict(vec)[0]

    confidence = round(max(proba) * 100, 2)

    if label == 1:
        return "REAL", confidence
    else:
        return "FAKE", confidence

def predict_news_voice():
    text = listen_news()
    if not text:
        return

    label, confidence = predict_text(text)

    result = f"This news is {label}. Confidence {confidence} percent."
    print("🔍 Result:", result)
    speak(result)

if __name__ == "__main__":
    predict_news_voice()
