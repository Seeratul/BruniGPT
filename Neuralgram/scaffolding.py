import torch
import numpy as np
import torch.nn.functional as f
import random

class FSBigramLM():
    def __init__(self, vocab_size, lr = 1e-3):
        #lh target of w rh source of w
        self.token_embedding_table = np.random.randn(vocab_size, vocab_size)
        self.lr = lr


    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers
        logits = [] 
        logits = np.matvec(self.token_embedding_table ,idx.T)
        #softmax because it makes it all easier
        logits =  np.clip(logits,1e-8,1e8)
        probs_us = np.exp(logits)
        logits = (probs_us/np.sum(probs_us))[0]
        #logits = logits.view(B*T,C)
        if targets is not None:
            loss = (-np.log(logits) * targets.T)
        else:
            loss = None
        return logits ,loss
    
    def generate(self,idx, max_new_tokens):
        #idx is (B,T) array
        for _ in range(max_new_tokens):
            keys = np.arange(0,10)
            #get the predictor
            logits,loss = self.forward(idx)
            #sample from dist
            print(logits.shape)
            print(keys.shape)
            idx_next = random.choices(keys, weights=logits, k=1)
            #append sampled index
            self.idx = np.concatenate((idx,idx_next),dim=1)
        return idx,logits

    def backwards(self,input,target,probs):
        #derivative softmax
        #print(probs)
        #print(target.T)
        dldx= probs-target.T

        #derivative activationfunction
        #No activation function :)

        #derivative inlast
        dldz = input

        #mul
        deltas=dldx*dldz
        deltas = deltas.T
        self.token_embedding_table = self.token_embedding_table - self.lr*deltas

        return None


class BigramLM(torch.nn.Module):
    loss = None
    def __init__(self, vocab_size):
        super(BigramLM, self).__init__()
        #lookup table?:( aaaa
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)
        B,T,C = logits.shape
        #logits = logits.view(B*T,C)
        if targets != None:
            targets = targets.view(B*T)
            logits = logits.view(B*T,C)
            loss = f.cross_entropy(logits,targets)
        else:
            loss = None
        return logits ,loss

    def generate(self,idx, max_new_tokens):
        #idx is (B,T) array
        for i in range(max_new_tokens):
            #get the predictor
            logits,loss = self(idx)
            #focus only on the last time step
            logits = logits[:,-1,:]
            #apply softmax 
            probs = f.softmax(logits, dim=-1)
            #sample from dist
            idx_next = torch.multinomial(probs,num_samples=1)
            #append sampled index
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

    def backwards():
        return None

class stoitos:
    def __init__(self,text):
        self.stoi = {ch:i for i,ch in enumerate(text)}
        self.itos = {i:ch for i,ch in enumerate(text)}

    def encode(self,s):
        return [self.stoi[c] for c in s]
    def decode(self,l):
        return [self.itos[i] for i in l]

def get_batch(enstr,block_size= 8,batch_size = 4):
    ix = torch.randint(len(enstr) - block_size, (batch_size,))
    x = torch.stack([enstr[i:i+block_size] for i in ix])
    y = torch.stack([enstr[i+1:i+block_size+1] for i in ix])
    return x, y


if __name__ == "__main__":
    n = FSBigramLM(10)
    starting_c = np.zeros((10,1))
    starting_c[7] = 1
    target = np.zeros((10,1))
    target[8] = 1

    out,loss =n.forward(starting_c,target)
    n.backwards(starting_c,target,out)
    print(out)
    #print(n.token_embedding_table[7])
    for _ in range(1000):
        out,loss =n.forward(starting_c,target)
        n.backwards(starting_c,target,out)
    print(out)
    #print(np.matvec(n.token_embedding_table,starting_c.T))



    
    #generated_c = n.generate(idx=starting_c,max_new_tokens=100)
    #print(generated_c)