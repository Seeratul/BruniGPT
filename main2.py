import Task1.bpe as bpe
import Task2.Ngram as ngram
import Task2.utils as utils


if __name__ == "__main__":
    k = 2000
    n = 3
    use_old = False
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    print("Files read")
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=k,extra_runtime=k-1)
    
    print("Vocab Setup Done")
    tt = bpe.tokenizetext(text,merge_rules)
    vtt = bpe.tokenizetext(vtext,merge_rules)
    print("Tokenization Done")
    modeln = ngram.y_grammodel(n,tt)
    modeln.train()
    print("Modelgeneration Done")
    perplexity = utils.evaluator(vtt,modeln,n)
    print("Perplexity: "+str(perplexity[0]))
    ctext = "shall i "
    ctt = bpe.tokenization_list(ctext,merge_rules)

    print(utils.sentencegen(tuple(ctt),modeln,1000,20))


