from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nlp import X_train_vec, y_train, X_test_vec, y_test

# Model klasyfikacji
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Przewidywanie i ocena modelu
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
