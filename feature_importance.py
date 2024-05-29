import numpy as np
from classification import model
from nlp import vectorizer

# Znaczenie cech
importance = np.abs(model.coef_[0])
feature_names = vectorizer.get_feature_names_out()

# Sortowanie cech według ich znaczenia
indices = np.argsort(importance)[::-1]
top_features = [feature_names[i] for i in indices[:10]]

print("Top 10 features:")
for feature in top_features:
    print(feature)
