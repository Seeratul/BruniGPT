import pickle
from collections import defaultdict
import Task1.bpe as bpe
import Task2.Ngram as ngram
import Task2.utils as utils
import numpy as np
import main4 as m4



def hpo(square = 5,lrmin=0.0001,lrmax=0.001,n_embdmin=8,n_embdmax=72):
    
    results = [[] for x in range(square+1)]
    runlenght = [[] for x in range(square+1)]
    etmin = 0
    et = 0
    results[square] = []
    runlenght[square] = []
    for i in range(1,square+1):
        results[i-1] = []
        runlenght[i-1]= []
        print("Outer Run "+str(i)+ " started")
        lr_i = lrmin+(lrmax-lrmin)/(square-1)*(i-1)
        results[i-1].append(lr_i)
        runlenght[i-1].append(lr_i)
        for j in range(1,square+1):
            print("Inner Run "+str(j)+ " of outer Run " +str(i)+ " started")
            n_embd_j=int(n_embdmin+(j-1)*(n_embdmax-n_embdmin)/(square-1))
            loss,rl = m4.Hpohook(lr = lr_i,n_embd=n_embd_j)
            results[i-1].append(loss)
            runlenght[i-1].append(rl)

            if i == square:
                results[i].append(n_embd_j)
                runlenght[i].append(n_embd_j)
    
    
    with open("gpttester2.pkl", "wb") as fp:
        pickle.dump([results,runlenght],fp)
    fp.close()

    return results,runlenght



if __name__ == "__main__":
    square = 5
    
    #results,runlenght = hpo(square=square,lrmin=1e-4,lrmax=1e-3,n_embdmin=8,n_embdmax=72)
    with open("gpttester2.pkl", "rb") as fp:
       [results,runlenght]=pickle.load(fp)
    fp.close()
    #results = [results]
    runlenght = [runlenght]
    results = runlenght
    out = ""
    for i in range(square):
       out = out +" "+str(results[0][square][i])
    print(str("lr ne")+out)
    for i in range(square):
        out = str("%.5f" % results[0][i][0])
        for j in range(1,square+1):
            out = out+ " "+str("%.2f" %  results[0][i][j])
        print(out)
    
           
