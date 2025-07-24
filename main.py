import pickle
from collections import defaultdict
import Preprocessing.bpe as bpe
import NNeighbours.Ngram as ngram
import numpy as np


def vocab_setup(text="",use_old=False):
    sample_text = text
    if(use_old):
        with open("vocab.pkl", "rb") as fp:
            final_vocab,merge_rules,vocabold=pickle.load(fp)
            fp.close()
    else:
        final_vocab, merge_rules,vocabold = bpe.preprocessing(sample_text,200)
        with open("vocab.pkl", "wb") as fp:
            pickle.dump([final_vocab,merge_rules,vocabold],fp)
        fp.close()
    return final_vocab, merge_rules,vocabold

def tokenizetext(text,merge_rules,use_old=False):
    if(use_old):
        with open("tl.pkl", "rb") as fp:
            tl=pickle.load(fp)
        fp.close()
        return tl
    else:
        text = bpe.tokenization_list(text,merge_rules)
        with open("tl.pkl", "wb") as fp:
            pickle.dump(tl,fp)
        fp.close()
        return text

def evaluator(text,ngram_model,n):
    n= n-1
    perplexity = 0
    probs = []

    test = True
    for i in range(len(text)-n):
        perplexity += np.log2(max(ngram_model.evaluate(tuple(text[i-n:i]),text[i]),0.00001))
        probs.extend([ngram_model[tuple(text[i-n:i])][text[i]]])
    mean_prob = np.mean(probs)
    perplexity = perplexity/i
    perplexity = np.power(2, -perplexity)
    return perplexity, mean_prob

def sentencegen(text,ngram_model,n):
    while len(text)<n:
        new = (modeln.generate(text),)
        text = text+new
    return text

if __name__ == "__main__":
    n = 7
    f = open("shakes.txt")
    text = f.read()
    f.close()
    final_vocab, merge_rules,vocabold = vocab_setup(text,use_old=False)
    print("Vocab Setup Done")
    tl = tokenizetext(text,merge_rules,use_old=False)
    print("Tokenization Done")
    print("compression rate "+ str(bpe.tokencounter(text)/len(tl)))
    modeln = ngram.y_grammodel(n,tl)
    modeln.train()
    print("Modelensemble Generated")
    #perplexity = evaluator(tl,modeln,n)
    #print("Evaluated!!:) Perplexity: "+ str(perplexity[0])+" Mean Prob:"+str(perplexity[1]))
    print(sentencegen(("never",),modeln,100))
    
    #print(ngram_model[('th', 'is</w>', 'is</w>')]["the"])
    #print(tl[0:3])
    #print(tuple(tl[0:3]))
   
