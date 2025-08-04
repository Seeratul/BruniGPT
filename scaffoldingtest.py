import Neuralgram.scaffolding as scaf
import torch
import Task1.bpe as bpe
import random
import numpy as np


if __name__ == "__main__":
    #Feeds pre bpe´d text in
    use_old = False
    f = open("sc_valid.txt")
    text = f.read()
    f.close()
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,use_old,n_merges=1000,extra_runtime=1000)
    tt = bpe.tokenizetext(text,("tl.pkl"),merge_rules,use_old)
    final_vocab.update(vocabold)
    trans= scaf.stoitos(final_vocab)
    tte=trans.encode(tt)
    x,y = scaf.get_batch(torch.tensor(tte))
    m = scaf.FSBigramLM(len(final_vocab))
    ll = []

     
    for steps in range(10000):
        i = random.randint(0,len(tte)-1)
        ohin =np.zeros((len(final_vocab),1))
        ohtar = np.zeros((len(final_vocab),1))
        ohin[tte[i]] = 1
        ohtar[tte[i+1]] = 1
        logitx,loss = m.forward(ohin,ohtar)
        m.backwards(ohin, ohtar,logitx)
        ll.append(max(loss[0]))
        if (steps%400==0):
            print(steps)
            print(min(ll))
    
    """
    starting_c = torch.zeros((1,1),dtype=torch.long)
    generated_c = m.generate(idx=starting_c,max_new_tokens=100)
    generated_c = generated_c[0].tolist()
    decoded_c = trans.decode(generated_c)
    print(decoded_c)
    """
    



    """
    optimzer = torch.optim.AdamW(m.parameters(),lr=1e-3)
    optimzer.zero_grad(set_to_none=True)
    batchsize = 32
    ll = []

    xb,yb = scaf.get_batch(torch.tensor(tte),batchsize)    
    for steps in range(10000):
        xb,yb = scaf.get_batch(torch.tensor(tte),batchsize)    
        logitx,loss = m(xb,yb)
        loss.backward()
        optimzer.step()
        ll.append(loss.item())
        if (steps%400==0):
            print(steps)
            print(min(ll))
    

    starting_c = torch.zeros((1,1),dtype=torch.long)
    generated_c = m.generate(idx=starting_c,max_new_tokens=100)
    generated_c = generated_c[0].tolist()
    decoded_c = trans.decode(generated_c)
    print(decoded_c)
    """
