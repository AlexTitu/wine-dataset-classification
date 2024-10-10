import numpy as np

# calculul prototipului unei clase
def prototype(data):
    mean = np.zeros(data[0].shape)
    for vector in data:
        mean += vector

    return mean/len(data)

# calculul matricii de covariatie
def covmatrix(data, mean_vector):
    cov = np.zeros((13, 13))
    for x in data:
        cov += np.matmul(np.atleast_2d(x-mean_vector).T, np.atleast_2d(x-mean_vector))
    return cov/len(data)

# calculul matricii K
def K_Matrix(cov, eigenNum): # (matrice covariatie, numar valori proprii retinute)

    pcaMatrix = []
    eigenValues, eigenVectors = np.linalg.eig(cov)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    # print(eigenValues)
    # print(eigenVectors[0])
    etha = 0

    if eigenNum is None: # daca nu se defineste numarul de valori proprii retinute
        for i in range(len(eigenValues)):
            etha += ((float(eigenValues[i])) / (float(np.sum(eigenValues)))) * 100.0
            print(etha)

            pcaMatrix.append(eigenVectors[i].tolist())
            if etha > 90.0: # se alege o matrice K ce retine peste 90% din informatia de baza
                return np.array(pcaMatrix)
    else: # altfel se retin atatea valori proprii cate au fost precizate
        for i in range(eigenNum):
            etha += ((float(eigenValues[i])) / (float(np.sum(eigenValues)))) * 100.0
            pcaMatrix.append(eigenVectors[i].tolist())
        print(etha)


    return np.array(pcaMatrix)

# aplicarea transformarii PCA
def PCA(data, K):
    # print(np.atleast_2d(data).shape)
    # print(K.shape)
    principalComponents = []
    for element in data:
        principalComponents.append(np.matmul(np.atleast_2d(element), K.T).flatten().tolist())

    return np.array(principalComponents)
