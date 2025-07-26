import pickle
from collections import defaultdict
import Preprocessing.bpe as bpe
import NNeighbours.Ngram as ngram
import TestingUtils.utils as utils
import numpy as np


def hpo(text,vtext,square = 5,merge_min=200,merge_max=2100,nmin=2,nmax=18,extra=True):
    
    results = [[] for x in range(square+1)]
    etmin = 0
    et = 0
    if(extra):
        etmin =merge_min
        et =merge_max
    results[square] = []
    for i in range(1,square+1):
        results[i-1] = []
        print("Outer Run "+str(i)+ " started")
        merges_i = int(merge_min+(merge_max-merge_min)/square*i)
        et_i =int(etmin+(et-etmin)/square*i)
        final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=merges_i,extra_runtime=et_i)
        tt = bpe.tokenizetext(text,("tl.pkl"),merge_rules)
        results[i-1].append(merges_i)
        vtt = bpe.tokenizetext(vtext,("vtl.pkl"),merge_rules)
        for j in range(1,square+1):
            print("Inner Run "+str(j)+ " of outer Run " +str(i)+ " started")
            n =int(nmin+j*(nmax-nmin)/square)
            modeln = ngram.y_grammodel(n,tt)
            modeln.train()
            perplexity = utils.evaluator(vtt,modeln,n)
            results[i-1].append(perplexity)
            if i == square:
                results[i].append(n)
    
    
    with open("ntester.pkl", "wb") as fp:
        pickle.dump([results],fp)
    fp.close()

    return results



if __name__ == "__main__":
    n = 4
    f = open("sd_train.txt")
    text = f.read()
    f.close()
    f = open("sd_valid.txt")
    vtext = f.read()
    f.close()
    results = hpo(text,vtext)

    print(results[0])
    print(results[1])
    print(results[2])
    print(results[3])
    print(results[4])
    print(results[5])
