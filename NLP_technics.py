import PyPDF2 as pdf
from PyPDF2 import PdfFileReader
import spacy

#   =======   Spacy Introduction  =======

nlp = spacy.load('en_core_web_sm')

doc = nlp("Apple isn't looking at buying UK startup for 1 billion dollars")

#tokenization
for token in doc:
    print(token.text, token.lemma_, token.is_stop)
print()

#part of speech tagging
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_)
print()

#named entity recognition
for ent in doc.ents:
    print(ent.text, ent.label_)
print()

#Sentence segmentation
for sent in doc.sents:
    print(sent)
print()
print()

doc1 = nlp("Welcome home! Please come here. Goodbye and good luck?")

for sent in doc1.sents:
    print(sent)

print()
print()


def set_rule(doc):
    for token in doc[:-1]:
        if token.text == '...':
            doc[token.i+1].is_sent_start = True
    return doc

nlp.add_pipe(set_rule, before='parser')
#nlp.remove_pipe(set_rule)

doc2 = nlp("Welcome home...Thanks for visiting...")

for sent in doc2.sents:
    print(sent)



