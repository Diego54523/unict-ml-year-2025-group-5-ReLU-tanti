import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

data = np.load(os.path.join(script_dir, "mri_features.npz"))
X_train = data['train_feats']
y_train = data['train_labels']
X_val = data['val_feats']
y_val = data['val_labels']

classes = data['classes'].tolist()

print("Train features loaded: ", X_train.shape)
print("Train labels loaded: ", y_train.shape)
print("Validation features loaded: ", X_val.shape)
print("Validation labels loaded: ", y_val.shape)
print("Loaded classes: ", classes)

# logistic regression

logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train, y_train)    

y_pred = logreg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
