import numpy as np

def evaluator(text,ngram_model,n):
    n= n-1
    perplexity = 0
    probs = []

    test = True
    for i in range(len(text)-n):
        perplexity += np.log2(max(ngram_model.evaluate(tuple(text[i-n:i]),text[i]),0.00001))
        probs.extend([ngram_model.evaluate(tuple(text[i-n:i]),text[i])])
    mean_prob = np.mean(probs)
    perplexity = perplexity/i
    perplexity = np.power(2, -perplexity)
    return perplexity, mean_prob

def sentencegen(text,modeln,n,top = 1):
    while len(text)<n:
        new = (modeln.generate_rand(text,top),)
        text = text+(new[0],)
    return text