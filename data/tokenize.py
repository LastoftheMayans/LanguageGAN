import os, nltk, os.path, sys, io

STOP = ["*stop*"]

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    elif not os.path.isdir(path):
        print "Error: " + str(path) + " is not a directory\n"
        sys.exit(1)

def tokenize_file(infile, outfile):
    try:
        tokenize_file_helper(infile, outfile)
    except UnicodeDecodeError:
        print "\tError tokenizing file " + str(infile) + " (UnicodeDecodeError)"
    except UnicodeEncodeError:
        print "\tError tokenizing file " + str(infile) + " (UnicodeEncodeError)"


def tokenize_file_helper(infile, outfile):
    # Read all input data
    with io.open(infile, 'r', encoding='utf8', errors='ignore') as f:
        data = f.read()

    # Try to decode into utf-8
    # data = data.decode('utf-8', 'ignore')

    # Strip all newlines
    data.replace('\n', '')

    # Break into lines
    output = nltk.sent_tokenize(data)

    # Tokenize lines
    output = [ nltk.word_tokenize(sentence) for sentence in output]

    # Recombine tokenized output
    output = unicode("\n".join([" ".join(sentence) for sentence in output]))

    # Write to output file
    with io.open(outfile, 'w', encoding='utf8') as f:
        f.write(output)


if __name__ == '__main__':
    # ensure target directory exists, create otherwise
    tokenized = os.path.normpath("tokenized")
    make_dir(tokenized)


    # loop through raw data
    original = os.path.normpath("original")
    for d in os.listdir("original"):
        path = os.path.join(original, d)
        target = os.path.join(tokenized, d)
        # ignore licenses directory and source.txt
        if os.path.isdir(path) and d != "licenses":
            # ensure target directory exists
            make_dir(target)
            # read 
            for f in os.listdir(path):

                print "Tokenizing file " + str(os.path.join(d, f))
                infile = os.path.join(path, f)
                outfile = os.path.join(target, f)
                tokenize_file(infile, outfile)

