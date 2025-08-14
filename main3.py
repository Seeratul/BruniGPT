import Task3.scaffolding as scaf
import torch
import Task1.bpe as bpe
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import Task3.utils as utils


if __name__ == "__main__": 
    #Intresting vars for the model:

    #Preset for a quick proof of concept run

    #After how many 400 iteration blocks without improvement in the best loss 
    #the program terminate
    patience = 3
   
    #How many merges BPE performs on the corpus
    k = 1
    #How many batches it trains for 
    trainingLenght = 2000
    #Learning rate
    lr = 1e-2

    #Presentation related setting:
    #Generate Example from random input
    GE = True
    #Create a plot for the Loss during Training
    Lossplot = True
    #Wheter to calculate perplexity on the validation set
    CalPe = True


    
    
    
    
    
    
    #Loads text and applies bpe

    trainingLenght_gt = trainingLenght
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=k,extra_runtime=k)
    tt = bpe.tokenizetext(text,merge_rules)
    final_vocab.update(vocabold)
    #Transforms the text into ints
    trans= scaf.stoitos(final_vocab)
    tte=trans.encode(tt)
    m = scaf.FSBigramLM(len(final_vocab),lr)
    ll = []
    
   
    minloss = 1000
    wait = 0
    for steps in range(trainingLenght):
        i = random.randint(0,len(tte)-1)
        context,target = scaf.get_batch(torch.tensor(tte),128,4)
        context = scaf.oneHot(context,len(final_vocab))
        target = scaf.oneHot(target,len(final_vocab))
        logitx,loss = m.forward_batch(context,target)
        m.backwards_batch(context,target,logitx)
        ll.append(loss[0])
        if (steps%400==0):  
            print(steps)
            print(sum(ll[steps-400:steps])/400)
            #Early stopping
            if (minloss > min(ll)):
                minloss = min(ll)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early Stopping")
                    trainingLenght_gt = steps
                    break


if GE:
    target = random.randint(0,len(final_vocab)-1)
    starting_c = np.zeros(len(final_vocab))
    starting_c[target] = 1
    generated_c,_ = m.generate(idx=starting_c,max_new_tokens=20)
    unhot_c = np.nonzero(generated_c)[1]
    decoded_c = trans.decode(unhot_c)
    print(decoded_c)
if Lossplot:
    lln = []
    h = 100
    for i in range(int(trainingLenght_gt/h)):
        lln.append(sum(ll[i*h:(i+1)*h])/h)
    p = f"Task3/saves/{lr}_{k}_{trainingLenght_gt}_{min(lln)}"
    pt = Path.cwd()/ p
    np.save(pt,m.token_embedding_table)
    plt.figure(figsize=(10, 5))
    plt.plot(lln)
    plt.title("Training Loss Over Steps")
    plt.xlabel(str(h)+"steps")
    plt.ylabel("Loss")
    plt.show()

if CalPe:
    f = open("sc_valid.txt")
    text = f.read()
    f.close()
    tt = bpe.tokenizetext(text,merge_rules)
    tte=trans.encode(tt)
    perplexity,mean_prob = utils.evaluator(tte,m,100,final_vocab)
    print("Perplexity: "+str(perplexity))
    print("Mean Probability: "+str(mean_prob))