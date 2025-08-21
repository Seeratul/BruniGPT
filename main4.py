import os
import json
import regex as re
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from Task1 import bpe
from Task3 import scaffolding as scaf
import pickle
import math
import time
from collections import defaultdict
import tiktoken
from Task2 import utils
import matplotlib.pyplot as plt
from pathlib import Path
from torcheval.metrics.text import Perplexity



class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for key, value in kwargs.items():
            setattr(self, key, value)

class CustomConfig(GPTConfig):
    n_layer = 4
    n_head = 8
    n_embd = 64
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    dropout = 0.1
    compile = False
    device = 'cpu'
    num_workers = 0
    max_iters = 1e5
    batch_size = 8
    block_size = 128
    learning_rate = 0
    betas = (0.9, 0.95)
    weight_decay = 1e-1
    grad_norm_clip = 1.0

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection 
    at the end.
    It's important in decoder block to have diagonal mask
    It is also possible to use torch.nn.MultiheadAttention.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.dropout = config.dropout
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(
                        torch.nn.functional, 
                        'scaled_dot_product_attention')
        if not self.flash:
            print(
              "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask", 
                torch.tril(torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size))
            
    def forward(self, x):
        # batch_size, seq_len, emb_dim
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # (b, seq_len, emb_dim) --> (b, seq_len, emb_dim * 3) --> (b, seq_len, emb_dim)
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (b, h, seq_len, d_k)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # (b, h, seq_len, d_k) matmul (b, h, d_k, seq_len) --> (b, h, seq_len, seq_len)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # diagonal mask
            # fill 0 mask with super small number so it wont affect the softmax weight
            # (batch_size, h, seq_len, seq_len)
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # (b, h, seq_len, seq_len) matmul (b, h, seq_len, d_k) --> (b, h, seq_len, d_k)
            y = att @ v 

        # (b, h, seq_len, d_k) --> (b, seq_len, h, d_k) --> (b, seq_len, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):
    # GPT decoder block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            act     = NewGELU(),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        
        # (batch_size, seq_len, emb_dim)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = "cpu"
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # positional token, shape (1, t)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        # (b, t, n_embd) -- > # (b, t, vocab_size)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        # -1 at output will be ignored
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None,stop_ids=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b, t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() in stop_ids:
                print("Early Stop")
                break
        return idx

class Trainer:

    def __init__(self, config, model,tester, train_dataset,patience=10):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.model = self.model.to(self.device)
        self.patience = patience
        self.pclock = 0
        self.minloss = 1000
        self.abort = False
        self.tl = []
        self.ll = []
        self.pp = []
        self.tester = tester

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

class Tester:

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.dataset = dataset
        self.callbacks = defaultdict(list)
        self.device = config.device
        self.model = self.model.to(self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def run(self):
        model, config = self.model, self.config

        # setup the dataloader
        loader = DataLoader(
            self.dataset,
            sampler=torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        self.iter_time = time.time()
        data_iter = iter(loader)
        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = [t.to(self.device) for t in batch]
        x, y = batch

        # forward the model
        logits, self.loss = model(x, y)
        #calc perplexity
        metric=Perplexity()
        metric.update(logits, y)
        perplexity = metric.compute().item()
        return self.loss,perplexity

class ShakespeareDataset(Dataset):
    def __init__(self, data, block_size=128, device_type='cpu'):
        self.block_size = block_size
        self.device_type = device_type
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)) 

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory().to('cuda', non_blocking=True)
        else:
            x, y = x.to('cpu'), y.to('cpu')
        return x, y

def batch_end_callback(trainer):
    trainer.ll.append(trainer.loss.item())
    if trainer.iter_num % 400 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: avg train loss {sum(trainer.ll[trainer.iter_num-400:trainer.iter_num])/400:.5f}")
        tel,perp = trainer.tester.run()
        trainer.tl.append(tel.item())
        trainer.pp.append(perp)
        print("test loss "+ str(trainer.tl[-1]))
        print("test perplexity "+ str(trainer.pp[-1]))
        if(trainer.tl[-1] < trainer.minloss):
            trainer.pclock = 0
            trainer.minloss = trainer.tl[-1]
        else:
            trainer.pclock +=1
            if (trainer.pclock>= trainer.patience):
                trainer.abort = True
                print("Early Stop")
                trainer.config.max_iters = trainer.iter_num

def Hpohook(lr=0.001,n_embd=64,depth= 8):
     #Vars
    k = 2000
    patience = 4
    #DataPrep
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    print("Files read")
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=k,extra_runtime=k)
    print("Vocab Setup Done")
    tt = bpe.tokenizetext(text,merge_rules)
    vtt = bpe.tokenizetext(vtext,merge_rules)
    print("Tokenization Done")
    final_vocab.update(vocabold)
    trans= scaf.stoitos(final_vocab)
    train_dataset=trans.encode(tt)
    test_dataset=trans.encode(vtt)
    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    #Expected Dataset shape:
    #At this point it expects a np.ndarray of ints
    #train_data = train_dataset
    #val_data = test_dataset
    #vocab_size = len(train_dataset)
    config = CustomConfig(vocab_size=len(final_vocab),learning_rate=lr,n_layer=depth,n_embd=n_embd)
    tconfig = CustomConfig(vocab_size=len(final_vocab),batch_size=32,block_size=64,learning_rate=0,n_embd=n_embd)


    #Create datasets and loader torch objects for train and ...
    train_dataset = ShakespeareDataset(train_dataset, config.block_size, config.device)
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,shuffle=True)
    #..for test
    test_dataset = ShakespeareDataset(test_dataset, tconfig.block_size, config.device)
    #test_loader = DataLoader(test_dataset, batch_size=tconfig.batch_size, num_workers=config.num_workers,shuffle=True)

    #Create model and set up trainer and Tester
    model = GPT(config).to(config.device)
    if config.compile:
        model = torch.compile(model)


    tester = Tester(tconfig, model, test_dataset)
    trainer = Trainer(config, model,tester, train_dataset,patience)
   


    #Defining batch end callback
   
    #Run
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    return min(trainer.tl),trainer.iter_num

if __name__ == "__main__":
    #Vars
    k = 0
    patience = 4
    pclock= 0
    lr = 5e-4
    n_embd = 72
    depth = 8
    #DataPrep
    f = open("sc_train.txt")
    text = f.read()
    f.close()
    f = open("sc_valid.txt")
    vtext = f.read()
    f.close()
    print("Files read")
    final_vocab, merge_rules,vocabold = bpe.vocab_setup(text,n_merges=k,extra_runtime=k)
    print("Vocab Setup Done")
    tt = bpe.tokenizetext(text,merge_rules)
    vtt = bpe.tokenizetext(vtext,merge_rules)
    print("Tokenization Done")
    final_vocab.update(vocabold)
    trans= scaf.stoitos(final_vocab)
    train_dataset=trans.encode(tt)
    test_dataset=trans.encode(vtt)
    train_dataset = np.array(train_dataset)
    test_dataset = np.array(test_dataset)

    #Expected Dataset shape:
    #At this point it expects a np.ndarray of ints
    train_data = train_dataset
    val_data = test_dataset
    #vocab_size = len(train_dataset)
    config = CustomConfig(vocab_size=len(final_vocab),learning_rate=lr,n_layer=depth,n_embd=n_embd)
    tconfig = CustomConfig(vocab_size=len(final_vocab),batch_size=32,block_size=64,learning_rate=0)


    #Create datasets and loader torch objects for train and ...
    train_dataset = ShakespeareDataset(train_dataset, config.block_size, config.device)
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,shuffle=True)
    #..for test
    test_dataset = ShakespeareDataset(test_dataset, tconfig.block_size, config.device)
    #test_loader = DataLoader(test_dataset, batch_size=tconfig.batch_size, num_workers=config.num_workers,shuffle=True)

    #Create model and set up trainer and Tester
    model = GPT(config).to(config.device)
    if config.compile:
        model = torch.compile(model)
    tester = Tester(tconfig, model, test_dataset)
    trainer = Trainer(config, model,tester, train_dataset,patience)

    ll = []
    tl = []
    minloss = 1000
   

    #Run
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()

    #Prep Stopwords
    stopvocab = [k for k in final_vocab if "." in k]
    stopvocab.extend([k for k in final_vocab if "?" in k])
    stopvocab.extend([k for k in final_vocab if "!" in k])
    stop_ids = trans.encode(stopvocab)
    #Test output
    text = 'Lord: Rise! My people, conquer the north!'
    text = bpe.tokenizetext(text,merge_rules)
    sample_ids = torch.Tensor(trans.encode(text)).long()
    sample_ids = torch.unsqueeze(sample_ids, 0).to(config.device)
    result = model.generate(sample_ids, max_new_tokens=1000, temperature=1, do_sample=True, top_k=None,stop_ids=stop_ids)
    print(trans.decode(result.detach().cpu().tolist()[0]))
    #Calc average loss for plot
    lln = []
    h = 400
    for i in range(int(config.max_iters/h)):
        lln.append(sum(trainer.ll[i*h:(i+1)*h])/h)

    #Save to disk
    #p = f"Task3/saves/{config.learning_rate}_{k}_{config.max_iters}_{min(lln)}"
    #pt = Path.cwd()/ p
    #torch.save(model,pt)
    #No saving and loading as code was thought using local objects which cannot be saved

    #Generate Plot

    x = range(int(config.max_iters/h))
    y1 = lln
    y2 = trainer.tl[:-1]
    y3 = trainer.pp[:-1]

    plt.plot(x, y1, label ='trainloss')
    plt.plot(x, y2, label ='punctual testloss')
    plt.legend()
    plt.title("Over Steps")
    plt.xlabel(str(h)+"steps")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(x, y3, label ='punctual perplexiy')
    plt.legend()
    plt.title(" Over Steps")
    plt.xlabel(str(h)+"steps")
    plt.ylabel("Loss")
    plt.show()