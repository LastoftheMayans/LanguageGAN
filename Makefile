
train:
	python3 ./model.py 0 1

load:
	python3 ./model.py 0 0

test:
	python3 ./model.py 1 0

tokenize:
	python ./data/tokenize.py

clean:
	rm -f ./data/tokenize.pyc
	rm -f ./cache/*
	rm -rf ./__pycache__
	rm -f ./output.txt
