# LanguageGAN
Generative Adversarial Network (GAN) to mimic the writing style of any given author. The author can be specified by editing the corresponding hyperparameter in model.py. To run the network, simply run "make test".

# Data
All data (original and tokenized) is stored in the data directory. Further information can be found in the source.txt file. Data was selected based on availability and distinctiveness of authors' writing styles. To add more works of writing, directories and untokenized data can be added to the data/original directory. With nltk installed, execute tokenize.py (python 2.7) from within the data directory. It will produce the tokenized output in the tokenized directory, and the author/book can be specified in the model and retrained. The corpus uses utf-8 encoding. Any characters that cannot be interpreted with utf-8 will be ignored.

# Tokenizer
The tokenizer is taken from python's nltk. The corpus is first split into sentences, then the sentences are split into tokens. The tokenized output contains one sentence per line, and any number of tokens separated by spaces.

# Corpus
The corpus reads in the tokenized data and generates a map of words to unique integers. Once initialized, the corpus can be polled for a new batch of data. Each time the corpus loops through the data, it shuffles the order of the sentences. A list of integers can be fed to the corpus, which will translate it back into English.
