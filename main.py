import pickle
from collections import defaultdict
import Preprocessing.bpe as bpe
import NGram.Ngram as ngram
import TestingUtils.utils as utils
import numpy as np


if __name__ == "__main__":
    n = 4
    use_old = False
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    print("Files read")
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,use_old,n_merges=4000,extra_runtime=4000)
    print("Vocab Setup Done")
    tt = bpe.tokenizetext(text,("tl.pkl"),merge_rules,use_old)
    vtt = bpe.tokenizetext(vtext,("vtl.pkl"),merge_rules,use_old)
    print("Tokenization Done")
    print("compression rate in train "+ str(bpe.tokencounter(text)/len(tt)))
    print("compression rate in vaild "+ str(bpe.tokencounter(vtext)/len(vtt)))
    modeln = ngram.y_grammodel(n,tt)
    modeln.train()
    print("Modelensemble Generated")
    perplexity = utils.evaluator(tt,modeln,n)
    print("Evaluated Train: Perplexity: "+ str(perplexity[0])+" Mean Prob:"+str(perplexity[1]))
    perplexity = utils.evaluator(vtt,modeln,n)
    print("Evaluated Valid: Perplexity: "+ str("%.2f" % perplexity[0])+" Mean Prob:"+str("%.2f" %perplexity[1]))
    print(utils.sentencegen(("i",),modeln,100,top=10)) 
    #print(modeln.probs(("con",)))
    #print(tl[0:3])
    #print(tuple(tl[0:3]))

