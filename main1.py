import Task1.bpe as bpe


if __name__ == "__main__":
    n = 2000
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    print("Files read")
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=n,extra_runtime=0)
    print("Vocab Setup Done")
    tt = bpe.tokenizetext(text,merge_rules)
    vtt = bpe.tokenizetext(vtext,merge_rules)
    print("Tokenization Done")
    print("Compression rate (nb) in train "+ str(bpe.tokencounter(text)/len(tt)))
    print("Compression rate (nb) in vaild "+ str(bpe.tokencounter(vtext)/len(vtt)))


