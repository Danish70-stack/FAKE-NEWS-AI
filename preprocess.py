import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download required resources
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

# customize stopwords (keep negations)
stop_words = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "nor", "never", "against"}
stop_words = stop_words - NEGATIONS

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not text:
        return ""

    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # keep numbers (important for real news)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)


# test
if __name__ == "__main__":
    sample = "Breaking News!!! Govt released 25% data on Jan 2024"
    print(clean_text(sample))
