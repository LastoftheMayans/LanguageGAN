
train:
	python3 ./model.py 0 0

load:
	python3 ./model.py 0 1

test:
	python3 ./model.py 1 1

tokenize:
	python ./data/tokenize.py

clean:
	rm -f ./data/tokenize.pyc
	rm -f ./gan_saved_model/*
