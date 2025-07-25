import collections
import json
from typing import List, Tuple, Dict
import random

def generate_ngram_model(n: int, samplett: List[str]) -> Dict[Tuple[str, ...], List[Tuple[str, int]]]:
    """
    Generates an n-gram model from a list of tokens.

    The model is a dictionary where each key is a tuple of (n-1) preceding words
    (the context), and the value is a list of tuples, each containing a
    possible following word and its frequency of occurrence.

    For example, with n=3 (trigrams), a key would be ('the', 'quick') and a
    value could be [('brown', 1)].

    Args:
        n: The size of the n-gram (e.g., 2 for bigrams, 3 for trigrams).
           Must be 1 or greater.
        tokens: A list of strings (words) from which to build the model.

    Returns:
        A dictionary representing the n-gram model. Returns an empty dictionary
        if not enough tokens are provided to create a single n-gram.
    """
    if n == 0:
        raise ValueError("n must be at least 1 to not cause havoc.")

    if len(samplett) < n:
        return {}  # Not enough tokens to create even one n-gram.

    # Use defaultdict with a Counter for efficient nested counting.
    # The key will be the context (a tuple of n-1 words), and the value
    # will be a Counter for the words that follow that context.
    ngram_counts = collections.defaultdict(collections.Counter)
    # Iterate through the tokens to build n-grams.
    # We stop at the point where the last full n-gram can be formed.
    for i in range(len(samplett) - n + 1):
        # The context is the tuple of n-1 words preceding the target word.
        context = tuple(samplett[i : i + n - 1])
        # The target word is the nth word in the sequence.
        following_word = samplett[i + n - 1]
        ngram_counts[context][following_word] += 1

    for i in (ngram_counts):
        total = 0
        for j in (ngram_counts[i]):
            total += (ngram_counts[i][j])+1
        for k in (ngram_counts[i]):
            ngram_counts[i][k] = (ngram_counts[i][k]+1)/total

    #fixing an edge case
    if n == 1:
        unigram_counts = {}
        unigram_counts[()]= ngram_counts[()]
        ngram_counts = unigram_counts
    return ngram_counts

class y_grammodel:
    def __init__(self,n: int, samplett):
        self.nm = n
        self.samplett = samplett
        self.decay = 0.5 
        #Todo mae decay a hyperparameter
        self.ngram_dict = [None]*(n)
        if self.nm<1:
            raise ValueError("n must be at least 1 to not cause havoc.")

    def train(self):
        for i in range(self.nm):
            self.ngram_dict[i] = generate_ngram_model(i+1,self.samplett)
        return True
    
    def probs(self,tokens):
        casedict = {}
        tl = len(tokens)
        if(tl>self.nm-1):
            print("Input is longer than n-1 meaning part of it will be ignored")
            tokens = tokens[tl-(self.nm-1):tl]
            tl = len(tokens)
        if(tl>= 3):
            for i in self.ngram_dict[tl-2][tokens[2:]]:
                casedict[i] = self.ngram_dict[tl][tokens][i]*0.6
                casedict[i] += self.ngram_dict[tl-1][tokens[1:]][i]*0.2
                casedict[i] += self.ngram_dict[tl-2][tokens[2:]][i]*0.1
            return casedict
        elif(tl==2):
            for i in self.ngram_dict[tl-2][()]:
                casedict[i] = self.ngram_dict[2][tokens][i]*0.6
                casedict[i] += self.ngram_dict[1][tokens[1:]][i]*0.2
                casedict[i] += self.ngram_dict[0][()][i]*0.1
            return casedict
        elif(tl==1):
            for i in self.ngram_dict[tl-1][()]:
                casedict[i] = self.ngram_dict[1][tokens][i]*0.6
                casedict[i] = self.ngram_dict[0][()][i]*0.3
            return casedict
        elif(tl==0):
            casedict = self.ngram_dict[tl][()]
            return casedict
        else:
            raise Exception('Something went horribly wrong in y_grammodel with your input.')
            
    def evaluate(self,tokens,target):
        probs = self.probs(tokens)
        return probs[target]
    
    def generate(self,tokens):
        probs = self.probs(tokens)
        return  max(probs, key=probs.get)
    
    def generate_rand(self,tokens,topn):
        probs = self.probs(tokens)
        total = 0
        probs2 = {}
        c = 0
        for i in probs:
            total += (probs[i])
            probs2[i]= probs[i]
            c+=1
            if c >= topn:
                break
        for j in probs2:
            probs2[j] = probs2[j]*(1/total)
        return  random.choices(list(probs2.keys()), weights=list(probs2.values()), k=1)[0]
       



# Example Usage
if __name__ == "__main__":
    # Sample text from "A Tale of Two Cities"
    sample_text = """
    It was the best of times, it was the worst of times,
    it was the age of wisdom, it was the age of foolishness,
    it was the epoch of belief, it was the epoch of incredulity,
    it was the season of Light, it was the season of Darkness,
    it was the spring of hope, it was the winter of despair.
    """

    # A more robust solution would use a proper tokenizer (e.g., from NLTK or spaCy)
    # and handle punctuation, but for this example, we'll just split by space.
    sample_tt = sample_text.lower().replace(',', '').replace('.', '').split()
    # For a bigram model, the key is a 1-word context.
    bigram_model = generate_ngram_model(n=3,samplett=sample_tt)
    # Convert tuple keys for the bigram model as well.
    #print(bigram_model)
  
