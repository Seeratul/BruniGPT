import numpy as np
import torch


def evaluator(text,model,n,gpt = False):
    n= n-1
    perplexity = 0
    probs = []
    if gpt:# This is for gpt
        for i in range(int(len(text)-n)):
            print(i)
            print(n)
            context = torch.Tensor(text[i-n:i]).long()
            context = torch.unsqueeze(context, 0).to("cpu")
            targets = torch.Tensor(text[i]).long()
            targets = torch.unsqueeze(targets, 0).to("cpu")
            perplexity += np.log2(max((model(idx = context,targets=targets)),1e-20))
            probs.extend([model(idx =(text[i-n:i]),targets=text[i])])
    else: # This for n gram
        for i in range(len(text)-n):
            perplexity += np.log2(max((model.evaluate(tuple(text[i-n:i]),text[i])),1e-20))
            probs.extend([model.evaluate(tuple(text[i-n:i]),text[i])])
    mean_prob = np.mean(probs)
    perplexity = perplexity/i
    perplexity = np.power(2, -perplexity)
    return perplexity, mean_prob

def sentencegen(text,modeln,n,top = 1):
    while len(text)<n:
        new = (modeln.generate_rand(text,top),)
        text = text+(new[0],)
    return text