import pickle
from collections import defaultdict
import Preprocessing.bpe as bpe
import NNeighbours.Ngram as ngram
import numpy as np

def vocab_setup(sample_text="",use_old=False,n_merges= 300,extra_runtime=2000):
    if(use_old):
        with open("vocab.pkl", "rb") as fp:
            final_vocab,merge_rules,vocabold=pickle.load(fp)
            fp.close()
    else:
        final_vocab, merge_rules,vocabold = bpe.preprocessing(sample_text,n_merges,extra_runtime)
        with open("vocab.pkl", "wb") as fp:
            pickle.dump([final_vocab,merge_rules,vocabold],fp)
        fp.close()
    return final_vocab, merge_rules,vocabold

def tokenizetext(text,vtext,merge_rules,use_old=False):
    if(use_old):
        with open("tl.pkl", "rb") as fp:
            tl=pickle.load(fp)
        fp.close()
        with open("vtl.pkl", "rb") as fp:
            vtl=pickle.load(fp)
        fp.close()
        return tl,vtl

    else:
        text = bpe.tokenization_list(text,merge_rules)
        with open("tl.pkl", "wb") as fp:
            pickle.dump(text,fp)
        fp.close()
        vtl = bpe.tokenization_list(vtext,merge_rules)
        with open("vtl.pkl", "wb") as fp:
            pickle.dump(vtl,fp)
        fp.close()
        
        return text,vtl

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

if __name__ == "__main__":
    n = 4
    use_old = True
    f = open("sd_train.txt")
    text = f.read()
    f.close()
    f = open("sd_valid.txt")
    vtext = f.read()
    f.close()
    final_vocab, merge_rules,vocabold = vocab_setup(text,use_old,n_merges=400,extra_runtime=400)
    print("Vocab Setup Done")
    tt,vtt = tokenizetext(text,vtext,merge_rules,use_old)
    print("Tokenization Done")
    print("compression rate in train "+ str(bpe.tokencounter(text)/len(tt)))
    print("compression rate in vaild "+ str(bpe.tokencounter(vtext)/len(vtt)))
    modeln = ngram.y_grammodel(n,tt)
    modeln.train()
    print("Modelensemble Generated")
    #perplexity = evaluator(tt,modeln,n)
    #print("Evaluated Train: Perplexity: "+ str(perplexity[0])+" Mean Prob:"+str(perplexity[1]))
    #perplexity = evaluator(vtt,modeln,n)
    #print("Evaluated Valid: Perplexity: "+ str(perplexity[0])+" Mean Prob:"+str(perplexity[1]))
    print(sentencegen(("i",),modeln,10,top=10)) 
    #print(modeln.probs(("con",)))
    #print(tl[0:3])
    #print(tuple(tl[0:3]))

