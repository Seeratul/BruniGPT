import numpy as np
import torch
import random
import Task3.scaffolding as scaf



def evaluator(text,model,k,final_vocab):
    n = 1
    perplexity = 0
    probs = []
    for i in range(k):
        i = random.randint(0,len(text)-1)
        context,target = scaf.get_batch(torch.tensor(text),128,4)
        context = scaf.oneHot(context,len(final_vocab))
        target = scaf.oneHot(target,len(final_vocab))
        logitx,loss = model.forward_batch(context,target)
        prob = (logitx * target).sum()/(128*4)
        prob = max(float(prob),1e-20)
        perplexity += np.log2(prob)
        probs.append(prob)

    mean_prob = np.mean(probs)
    perplexity = perplexity/k
    perplexity = np.power(2, -perplexity)
    return perplexity, mean_prob

def sentencegen(text,modeln,n,top = 1):
    eost = [".","?","!"]
    while len(text)<n:
        new = (modeln.generate_rand(text,top),)
        text = text+(new[0],)
        for i in eost:
            if i in text:
                return text
    return text

def oneHot(tensor,vocab_size):
    batch,block = tensor.shape
    tout = np.zeros((batch,block,vocab_size))
    for i in range(batch):
        for j in range(block):
            tout[i,j,tensor[i,j]] = 1
    return tout

