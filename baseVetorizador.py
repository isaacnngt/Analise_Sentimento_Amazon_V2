from gerenciador import Gerenciador
from vetorizador import CountVectorizer, TFIDFVectorizer

gerenciador = Gerenciador()
revisoes = gerenciador.revisoes
recomendacoes = gerenciador.recomendacoes

vetorizador1 = CountVectorizer(revisoes)
vetorizador2 = TFIDFVectorizer(revisoes)

print(len(vetorizador1.vetorizador.vocabulary_))
print(len(vetorizador2.vetorizador.vocabulary_))