import torch
import torch.nn.functional as f


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
        print(logits.shape)
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


