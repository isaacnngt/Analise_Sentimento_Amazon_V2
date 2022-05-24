from classificador import NaiveBayes, SVM

classificador1 = NaiveBayes("neive_bayes_1_1.pickle")
classificador2 = SVM("SVM_1_1.pickle")

print(classificador1.marcador())
print(classificador2.marcador())