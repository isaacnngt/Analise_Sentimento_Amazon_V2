from gerenciador import Gerenciador
from vetorizador import CountVectorizer, TFIDFVectorizer
from classificador import NaiveBayes, SVM

gerenciador = Gerenciador()
revisoes = gerenciador.revisoes
recomendacoes = gerenciador.recomendacoes

vetorizador1 = CountVectorizer(revisoes)
vetorizador2 = TFIDFVectorizer(revisoes)

classificador1 = NaiveBayes("neive_bayes_1_1.pickle",vetorizador1, recomendacoes)
classificador2 = SVM("SVM_1_1.pickle",vetorizador2, recomendacoes)
