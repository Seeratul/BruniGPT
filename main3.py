import Task3.scaffolding as scaf
import torch
import Task1.bpe as bpe
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__": 
    #Loads text and applies bpe
    n = 1
    trainingLenght = 40000
    f = open("sc_test.txt")
    text = f.read()
    f.close()
    #text = text[0:9]
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=n,extra_runtime=n)
    tt = bpe.tokenizetext(text,merge_rules)
    final_vocab.update(vocabold)
    #Transforms the text into ints
    trans= scaf.stoitos(final_vocab)
    tte=trans.encode(tt)
    m = scaf.FSBigramLM(len(final_vocab))
    ll = []
    #context,target = scaf.get_batch(torch.tensor(tte),8,4)
    
   
     
    for steps in range(trainingLenght):
        i = random.randint(0,len(tte)-1)
        context,target = scaf.get_batch(torch.tensor(tte),8,4)
        context = scaf.oneHot(context,len(final_vocab))
        target = scaf.oneHot(target,len(final_vocab))
        logitx,loss = m.forward_batch(context,target)
        m.backwards_batch(context,target,logitx)
        ll.append(loss[0])
        if (steps%400==0):  
            print(steps)
            print(min(ll))
    
#Broken I-O code.
#    starting_c = torch.zeros((1,1),dtype=torch.long)
#    generated_c = m.generate(idx=starting_c,max_new_tokens=100)
#    generated_c = generated_c[0].tolist()
#    decoded_c = trans.decode(generated_c)
#    print(decoded_c)

    lln = []
    h = 100
    for i in range(int(trainingLenght/h)):
        lln.append(sum(ll[i*h:(i+1)*h])/h)


    plt.figure(figsize=(10, 5))
    plt.plot(lln)
    plt.title("Training Loss Over Steps")
    plt.xlabel(str(h)+"steps")
    plt.ylabel("Loss")
    #plt.grid(True)
    plt.show()