from sklearn.model_selection import cross_val_score
from nlp import vectorizer, data
from feature_importance import model


# Walidacja krzyżowa
scores = cross_val_score(model, vectorizer.transform(data['review']), data['sentiment'], cv=5)
print(f'Cross-Validation Accuracy Scores: {scores}')
print(f'Mean CV Accuracy: {scores.mean()}')
