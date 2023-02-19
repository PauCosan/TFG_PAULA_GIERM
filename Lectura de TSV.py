import csv
import codecs

with codecs.open('AccidentsTweets.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    for fila in lector_tsv:
        # Accede a las columnas de cada fila
        for columna in fila:
            # Imprime el contenido de cada columna
            print(columna)
