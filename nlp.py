from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('drugsComTrain_raw.tsv', sep='\t', header=0)
#print(data)
# Filtruj potrzebne kolumny
data = data[['review', 'rating']]
data.dropna(inplace=True)
print(data)

# Przypisanie labeli: rating >= 7 to pozytywne, jak nie to negatywne
data['sentiment'] = data['rating'].apply(lambda x: 1 if x >= 7 else 0)
# Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=420)

# Wektoryzacja tekstu
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
print(X_train_vec)
X_test_vec = vectorizer.transform(X_test)
