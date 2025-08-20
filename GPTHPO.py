import pickle
from collections import defaultdict
import Task1.bpe as bpe
import Task2.Ngram as ngram
import Task2.utils as utils
import numpy as np
import main4 as m4



def hpo(square = 5,lrmin=0,lrmax=4500,depthmin=0,depthmax=1):
    
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
            depth_j=int(depthmin+(j-1)*(depthmax-depthmin)/(square-1))
            loss,rl = m4.Hpohook(lr_i,depth_j)
            results[i-1].append(loss)
            runlenght[i-1].append(rl)

            if i == square:
                results[i].append(depth_j)
                runlenght[i].append(depth_j)
    
    
    with open("gpttester.pkl", "wb") as fp:
        pickle.dump([results,runlenght],fp)
    fp.close()

    return results,runlenght



if __name__ == "__main__":
    square = 5
    
    #[results,runlenght] = [hpo(square=square,lrmin=1e-5,lrmax=1e-2,depthmin=2,depthmax=12)]
    with open("gpttester.pkl", "rb") as fp:
       [results,runlenght]=pickle.load(fp)
    fp.close()
    results = [runlenght]
    #runlenght = [runlenght]
   
    out = ""
    for i in range(square):
       out = out +" "+str(results[0][square][i])
    print(str("k n")+out)
    for i in range(square):
        out = str("%.3f" % results[0][i][0])
        for j in range(1,square+1):
            out = out+ " "+str("%.2f" %  results[0][i][j])
        print(out)
    
           
