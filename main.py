# from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.datasets import load_wine
import time


# preluarea setului de date de la uci - trecut la sklearn
# wine = fetch_ucirepo(id=109)
data = load_wine()

# data (as pandas dataframes)
# X = wine.data.features
X = data.data
# y = wine.data.targets
y = data.target

# metadatele bazei de date uci
# print(wine.metadata)
counts, bins = np.histogram(y)

plt.figure('Histograma Claselor'), plt.hist(bins[:-1], bins, weights=counts), plt.title("Histograma Claselor")
# informatiile variabilelor
# print(wine.variables)
# 178 de vectori - vinuri
# print(X.shape) # 13 caracteristici ale vinului
# print(y.shape) # 1 eticheta pentru fiecare vin - 3 clase (tipuri) de vinuri

# datele de intrare convertite ca numpy array
Xnp = np.array(X)

# calculul matricii de covariatie
# print(Xnp)
# print(prototype(Xnp))

# implementare proprie
# cov1 = covmatrix(Xnp, prototype(Xnp))
# implementare numpy
cov2 = np.cov(Xnp, rowvar=False, bias=True)

# print(cov1.shape)
# print(cov2.shape)
# print(covmatrix(Xnp, prototype(Xnp)))
# print(np.cov(X, rowvar=False))
# print(abs(cov1 - cov2))

# evaluarea algoritmilor in functie de numarul de valori proprii retinute

for numValues in range(1, 14):
    K = PCA.K_Matrix(cov2, numValues)  # calcul matrice K
    # print(K)
    # print(K.shape)
    # print(Xnp.shape)
    X_comp = PCA.PCA(Xnp, K)
    Y_comp = np.array(y)
    # print(X_comp.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X_comp, Y_comp, test_size=0.3, random_state=19, stratify=Y_comp)

    NN1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    NN3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    K_Means_supervizat = NearestCentroid()

    t1 = time.time()
    NN1.fit(X_train, np.ravel(Y_train))
    t2 = time.time()
    print(f"Time Train NN: {t2 - t1}")

    t1 = time.time()
    NN3.fit(X_train, np.ravel(Y_train))
    t2 = time.time()
    print(f"Time Train 3-NN: {t2 - t1}")

    t1 = time.time()
    K_Means_supervizat.fit(X_train, np.ravel(Y_train))
    t2 = time.time()
    print(f"Time Train K-Means: {t2 - t1}")

    print(f"Pentru {numValues} valori proprii")
    t1 = time.time()
    print(f"Scor clasificare NN:{NN1.score(X_test, np.ravel(Y_test))}")
    t2 = time.time()
    print(f"Time Test NN: {t2-t1}")
    t1 = time.time()
    print(f"Scor clasificare 3-NN:{NN3.score(X_test, np.ravel(Y_test))}")
    t2 = time.time()
    print(f"Time Test 3-NN: {t2 - t1}")
    t1 = time.time()
    print(f"Scor clasificare K-Means Supervizat: {K_Means_supervizat.score(X_test, np.ravel(Y_test))}")
    t2 = time.time()
    print(f"Time Test K-Means: {t2 - t1}")

# calculul matricii PCA si aplicarea transformarii
K = PCA.K_Matrix(cov2, 3)  # calcul matrice K
# print(K)
# print(K.shape)
# print(Xnp.shape)
X_comp = PCA.PCA(Xnp, K)
Y_comp = np.array(y)
# print(X_comp.shape)

# impartirea setului de date
X_train, X_test, Y_train, Y_test = train_test_split(X_comp, Y_comp, test_size=0.3, random_state=19, stratify=Y_comp)

# definirea clasificatorilor
NN1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
NN3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
K_Means_supervizat = NearestCentroid()

# antrenarea si testarea algoritmilor
NN1.fit(X_train, np.ravel(Y_train))
NN3.fit(X_train, np.ravel(Y_train))
K_Means_supervizat.fit(X_train, np.ravel(Y_train))

print(f"Pentru {3} valori proprii")
print(f"Scor clasificare NN:{NN1.score(X_test, np.ravel(Y_test))}")
print(f"Scor clasificare 3-NN:{NN3.score(X_test, np.ravel(Y_test))}")
print(f"Scor clasificare K-Means Supervizat: {K_Means_supervizat.score(X_test, np.ravel(Y_test))}")

# testarea algoritmilor si afisarea predictiilor facute
Y_1 = NN1.predict(X_test)
Y_3 = NN3.predict(X_test)
Y_K = K_Means_supervizat.predict(X_test)

plt.figure('NN'), plt.plot(X_test, Y_1, 'or'), plt.plot(X_test, Y_test, '^g'), plt.ylabel("Clasa"),
plt.xlabel("Valoare intrare"), plt.legend(["NN Predictii", "Real"]), plt.title("Esantioane NN-1")
plt.figure('3-NN'), plt.plot(X_test, Y_3, 'or'), plt.plot(X_test, Y_test, '^g'), plt.ylabel("Clasa"),
plt.xlabel("Valoare intrare"), plt.legend(["3-NN Predictii", "Real"]), plt.title("Esantioane NN-3")
plt.figure('K-Means Supervizat'), plt.plot(X_test, Y_K, 'or'), plt.plot(X_test, Y_test, '^g'), plt.ylabel("Clasa"),
plt.xlabel("Valoare intrare"), plt.legend(["K-Means Supervizat - Predictii", "Real"]),
plt.title("Esantioane K-Means Supervizat")

# calculul si afisarea matricii de confuzie pentru fiecare algoritm
conf_NN1 = confusion_matrix(Y_test, Y_1)
conf_NN3 = confusion_matrix(Y_test, Y_3)
conf_K = confusion_matrix(Y_test, Y_K)

disp_1 = ConfusionMatrixDisplay(confusion_matrix=conf_NN1, display_labels=data.target_names)
disp_1.plot()
plt.title("Matrice de Confuzie NN-1")
disp_3 = ConfusionMatrixDisplay(confusion_matrix=conf_NN3, display_labels=data.target_names)
disp_3.plot()
plt.title("Matrice de Confuzie NN-3")
disp_K = ConfusionMatrixDisplay(confusion_matrix=conf_K, display_labels=data.target_names)
disp_K.plot()
plt.title("Matrice de Confuzie K-Means Supervizat")
plt.show()
