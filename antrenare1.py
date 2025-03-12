import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score

# Presupunem că 'X' sunt datele preprocesate și 'y' sunt etichetele
X = np.random.rand(1500, 100)  # Exemplu de date random
y = np.random.randint(20, size=1000)  # Exemplu de etichete random

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def plot_roc_curve(y_test, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Model SVM
svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("Acuratețea modelului SVM:", accuracy_score(y_test, y_pred_svm))
plot_confusion_matrix(y_test, y_pred_svm, "Confusion Matrix - SVM")
plot_roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1], "SVM")

# Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Acuratețea modelului Random Forest:", accuracy_score(y_test, y_pred_rf))
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest")
plot_roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1], "Random Forest")

# Model Logistic Regression
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
print("Acuratețea modelului de regresie logistică:", accuracy_score(y_test, y_pred_logreg))
plot_confusion_matrix(y_test, y_pred_logreg, "Confusion Matrix - Logistic Regression")
plot_roc_curve(y_test, logreg_model.predict_proba(X_test)[:, 1], "Logistic Regression")
