import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Dataset sintético: [SRM, IBU]
X = np.array([
    [15,20],[12,15],[18,25],[14,17],  # Lager
    [45,20],[40,61],[42,70],[50,22],  # Stout
    [30,60],[28,65],[32,70],[27,55],  # IPA
    [25,25],[28,20],[30,22],[27,18]   # Scottish
])

# Etiquetas: 0=Lager, 1=Stout, 2=IPA, 3=Scottish
y = np.array([
    0,0,0,0,
    1,1,1,1,
    2,2,2,2,
    3,3,3,3
])

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# One vs. Rest (OVR)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=200))
ovr.fit(X_train, y_train)
y_pred_ovr = ovr.predict(X_test)

# One vs. One (OVO)
ovo = OneVsOneClassifier(LogisticRegression(max_iter=200))
ovo.fit(X_train, y_train)
y_pred_ovo = ovo.predict(X_test)

# Softmax (multinomial)
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
softmax.fit(X_train, y_train)
y_pred_softmax = softmax.predict(X_test)

# Evaluación y matriz de confusión
def evaluar(y_test, y_pred, titulo):
    print(f"\n--- {titulo} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Lager", "Stout", "IPA", "Scottish"]))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Lager", "Stout", "IPA", "Scottish"]).plot(cmap="Blues")
    plt.title(titulo)
    plt.show()

# Evaluar los tres modelos
evaluar(y_test, y_pred_ovr, "OVR")
evaluar(y_test, y_pred_ovo, "OVO")
evaluar(y_test, y_pred_softmax, "Softmax multinomial")