import pickle
from collections import defaultdict
import Task1.bpe as bpe
import Task2.Ngram as ngram
import Task2.utils as utils
import numpy as np


def hpo(text,vtext,square = 10,merge_min=0,merge_max=5000,nmin=0,nmax=10,extra=True):
    
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
        merges_i = int(merge_min+(merge_max-merge_min)/(square-1)*(i-1))
        et_i =int(etmin+(et-etmin)/(square-1)*(i-1))
        final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=merges_i,extra_runtime=et_i)
        tt = bpe.tokenizetext(text,merge_rules)
        results[i-1].append(merges_i)
        vtt = bpe.tokenizetext(vtext,merge_rules)
        for j in range(1,square+1):
            print("Inner Run "+str(j)+ " of outer Run " +str(i)+ " started")
            n =int(nmin+(j-1)*(nmax-nmin)/(square-1))
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
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    square = 10
    results = [hpo(text,vtext,square=square,merge_min=0,merge_max=5000,nmin=1,nmax=10,extra=True)]
    #with open("ntester.pkl", "rb") as fp:
    #   results=pickle.load(fp)
    #fp.close()
   
   
    for i in range(square):
        out = str(results[0][i][0])
        for j in range(1,square+1):
            out = out+ " "+str("%.2f" % results[0][i][j][0])
        print(out)
    out = ""
    for i in range(square):
       out = out +" "+str(results[0][square][i])
    print("x "+out)
           
