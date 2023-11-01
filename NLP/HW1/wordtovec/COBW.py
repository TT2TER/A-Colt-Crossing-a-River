from gensim.models import Word2Vec
import nltk
from nltk.corpus import brown

nltk.download('brown')
data = brown.sents()  # Use the Brown corpus from NLTK as sample data

model = Word2Vec(data, min_count=1, size=100, window=5, sg=0)


model.train(data, total_examples=len(data), epochs=200)
