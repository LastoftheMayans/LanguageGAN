import os, os.path, random
import numpy as np

class Corpus(object):
    def __init__(self, author, book=None, batch_size=1, sentence_length=12, stop="*stop*"):
        self.author = author
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.stop = stop

        self.books = self.load_titles(book)

        self.words = []
        self.vocabulary = {}
        self.corpus = self.load_corpus()
        self.vocab_size = len(self.words)

        self.index = 0
        self.size = len(self.corpus)

    def __len__(self):
        return self.size

    def load_titles(self, book):
        books = []

        # validate author
        auth_dir = os.path.join("data", "tokenized", self.author)
        if not os.path.exists(auth_dir):
            print("Error: no such author (" + author + ")")
            return None
        elif not os.path.isdir(auth_dir):
            print("Error: expected directory (" + str(auth_dir) + ")")
            return None


        # if no book is specified, load them all (validate that at least one is loaded)
        if book == None:
            count = 0
            for f in os.listdir(auth_dir):
                books.append(os.path.join(auth_dir, f))
                count = count + 1
            if count == 0:
                print("Error: no books available for author " + author)
                return None
        # otherwise validate the book
        else:
            b = os.path.join(auth_dir, book)
            if not os.path.exists(b):
                print("Error: no such book (" + book + ")")
                return None
            elif not os.path.isfile(b):
                print("Error: expected file (" + str(b) + ")")
                return None
            books.append(b)

        return books

    def load_corpus(self):
        if len(self.vocabulary) > 0:
            print("Error: loading additional information into the corpus is not currently supported.")
            return None

        corpus = []
        # pull sentences from files
        for title in self.books:
            with open(title, 'r') as f:
                corpus = corpus + [ self.clip_sentence(sentence.lower().split(' ')) for sentence in f.read().split('\n') ]

        # create a list of unique words (such that self.stop is the first element -> 0)
        self.words = set([ word for sentence in corpus for word in sentence ])
        self.words.remove(self.stop)
        self.words = [ self.stop ] + list(self.words)
        self.vocabulary = { self.words[index]: index for index in range(len(self.words))}
        corpus = [ [ self.vocabulary[word] for word in sentence ] for sentence in corpus ]

        # shuffle the sentences together
        random.shuffle(corpus)

        return corpus

    def clip_sentence(self, sentence):
        return [ sentence[i] if i < len(sentence) else self.stop for i in range(self.sentence_length-1) ] + [ self.stop ]

    def next_batch(self):
        if(self.index + self.batch_size >= self.size):
            out1 = self.corpus[self.index:]
            self.index = self.index + self.batch_size - self.size
            out2 = self.corpus[:self.index-1]
            out = out1 + out2
            print("Warning: exceeding corpus bounds, data will loop\n")
        else:
            out = self.corpus[self.index:self.index+self.batch_size]
            self.index += self.batch_size
        return np.array(out).astype(int)


    def translate(self, sentence):
        # turn ints to words
        tokens = [ self.words[word] if word > 0 else '' for word in sentence ]

        # combine the list into a single sentence
        sentence = " ".join(tokens)

        # replace double apostrophes with quotes
        sentence = sentence.replace('\'\'', '"')

        # replace extended whitespace with smaller whitespace
        sentence = sentence.replace('  ', ' ')

        # remove whitespace in front of most punctuation (not quotes b/c of ambiguity)
        sentence = sentence.replace(' ,',',').replace(' .','.').replace(' !','!')
        sentence = sentence.replace(' ?','?').replace(' :',':').replace(' \'', '\'')

        return sentence
