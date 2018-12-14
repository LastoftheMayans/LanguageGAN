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

    # len is number of lines in the corpus
    # for size of vocabulary, use Corpus.vocab_size
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
        # at a later point, may allow adding additional books to the corpus
        if len(self.vocabulary) > 0:
            raise AssertionError("Cannot call Corpous.load_corpus outside of __init__")

        corpus = []
        # pull sentences from files
        for title in self.books:
            with open(title, 'r') as f:
                corpus = corpus + [ self.clip_sentence(sentence.lower().split(' ')) for sentence in f.read().split('\n') ]

        # create a list of unique words (such that self.stop is the first element -> 0)
        self.words = set([ word for sentence in corpus for word in sentence ])
        self.words.remove(self.stop)
        self.words = [ self.stop ] + list(self.words)

        # map each word to an int
        self.vocabulary = { self.words[index]: index for index in range(len(self.words))}

        # translate each word in the corpus to the corresponding int
        corpus = [ [ self.vocabulary[word] for word in sentence ] for sentence in corpus ]

        # remove all one-token and two-token sentences
        corpus = [ sentence for sentence in corpus if sentence[1] > 0 and sentence[2] > 0 ]

        # shuffle the sentences together
        random.shuffle(corpus)

        return corpus

    # remove any leading and trailing underscores (italics)
    def preprocess_word(self, word):
        if len(word) < 2:
            return word
        if word[0] == '_':
            word = word[1:]
        if word[-1] == '_':
            word = word[:-1]
        return word

    # set sentence to fixed length (sentence), and remove any leading or trailing numbers (most likely line numbers)
    def clip_sentence(self, sentence):
        sentence = sentence[1:] if sentence[0].strip().isdigit() else sentence
        sentence = sentence[:-1] if sentence[-1].strip().isdigit() else sentence
        sentence = [ self.preprocess_word(sentence[i]) if i < len(sentence) else self.stop for i in range(self.sentence_length-1) ] + [ self.stop ]
        return sentence

    # generate another batch of data (loop if needed)
    def next_batch(self):
        # test if we need to overflow
        if(self.index + self.batch_size >= self.size):
            # pull the last elements from the current list 
            out1 = self.corpus[self.index:]

            # randomize the corpus
            random.shuffle(corpus)

            # pull the rest of the data from the now-shuffled corpus
            self.index = self.index + self.batch_size - self.size
            out2 = self.corpus[:self.index]

            # combine
            out = out1 + out2

        else:
            out = self.corpus[self.index:self.index+self.batch_size]
            self.index += self.batch_size

        # enforce that the array contains only ints
        return np.array(out).astype(int)


    # given an array of word ints, turn it into a sentence string
    def translate(self, sentence):
        # turn ints to words
        tokens = [ self.words[word] for word in sentence if word > 0]

        # combine the list into a single sentence
        sentence = " ".join(tokens)

        # construct quotes
        sentence = sentence.replace(' \'\'', '"')
        sentence = sentence.replace('`` ', '"')

        # remove whitespace in front of most punctuation
        sentence = sentence.replace(' ,',',').replace(' .','.').replace(' !','!')
        sentence = sentence.replace(' ?','?').replace(' :',':').replace(' \'', '\'')

        return sentence
