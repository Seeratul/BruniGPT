import pickle
from collections import defaultdict

def tokenization_dict(text):
    """
    Tokenizes a raw text string into a dictionary of word counts.
    Args:
        text: The raw input string to be tokenized.

    Returns:
        A dictionary where keys are the words (tokens) and values are their
        frequencies.
    """
    # 1. Set all letters to lower case.
    lower_text = text.lower()

    # 2. Cut the string at every space.
    words = lower_text.split()

    # 3. Create a dictionary of word counts using base Python.
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts

def tokenization_list(text,merge_rules):
    change = True
    merge_dict = defaultdict(list)
    #split and set lower case
    words = text.lower().split()
    #tokenize leaving an artifact list
    textlist = [list(word+" ") for word in words]
    #removing artifact list
    textlist = [x for xs in textlist for x in xs]
    textlist = [w.replace(' ', '</w>') for w in textlist]

    for i in range(len(merge_rules)):
        merge_dict[merge_rules[i][0]].append(merge_rules[i][1])
    
    while change :
        f = 0
        change = False
        for i in range(len(textlist)):
            for j in range(len(merge_dict[textlist[i-f]])):
                if textlist[i+1-f]==merge_dict[textlist[i-f]][j]:
                    change = True
                    textlist[i-f] = textlist[i-f]+textlist[i+1-f]
                    textlist.pop(i+1-f)
                    f+= 1
                    break
                    
    return textlist

def get_stats(vocab):
    """Count frequencies of adjacent pairs in the vocabulary."""
    symboln = {}
    pairs = {}
    for word, freq in vocab.items():

        symbols = word.split()

        for i in range(len(symbols) -1):
            symboln[symbols[i]] = symboln.get(symbols[i], 0) + freq 
            pair = (symbols[i], symbols[i+1])
            pairs[pair] = pairs.get(pair, 0) + freq
        symboln[symbols[len(symbols)-1]] = symboln.get(symbols[len(symbols)-1], 0) + freq
       
    return pairs, symboln

def merge_vocab(pair, v_in):
    """Merge the most frequent pair in the vocabulary keys."""
    v_out = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in v_in.items():
        # Replace the pair in the word string
        new_word = word.replace(bigram, replacement)
        v_out[new_word] = freq
    return v_out

def bpe(word_counts, num_merges,frac = 0):
    """
    Performs Byte-Pair Encoding to learn merge rules and create a vocabulary.

    Args:
        word_counts (dict): A dictionary of words and their frequencies.
        num_merges (int): The number of merge operations to perform.
        frac(float): The fraction of total tokens that occur in this specific touple

    Returns:
        tuple: A tuple containing:
            - set: The final vocabulary of tokens.
            - list: The learned merge rules in order.
            - set: The starting vocab.
    """
    # 1. Prepare initial vocabulary from word counts
    #    - Split words into characters and add a special end-of-word token '</w>'
    vocab = {' '.join(list(word) + ['</w>']): count for word, count in word_counts.items()}
    vocabold = vocab
    merges = []
    for i in range(num_merges):
        # 2. Get statistics of adjacent pairs
        stats, symboln = get_stats(vocab)
        if not stats:
            raise ValueError('Your merge number leads back to the inital dataset.')
    

        # 3. Find the most frequent pair
        best_pair = max(stats, key=stats.get)
        if(stats[best_pair]>= symboln[best_pair[0]]*frac):
            merges.append(best_pair)
            # 4. Merge the pair in the vocabulary
            vocab = merge_vocab(best_pair, vocab)
        elif(stats[best_pair]>= symboln[best_pair[1]]*frac):
            merges.append(best_pair)
            # 4. Merge the pair in the vocabulary
            vocab = merge_vocab(best_pair, vocab)
        else:
            print("A tototal of "+ str(i-1) +" merges were performed before the frac threshold was reached")
            break
    print("All merges performed")

    # 5. Create the final token vocabulary from the keys of the learned vocab
    final_tokens = set()
    startingvocab = set()
    for word in vocab.keys():
        final_tokens.update(word.split(' '))

    for word in vocabold.keys():
        startingvocab.update(word.split(' '))

    return final_tokens, merges,startingvocab

def preprocessing(string,num_merges,frac=0):
    """
    Performs Byte-Pair Encoding to learn merge rules and create a vocabulary, includes text preprocessing.

    Args:
        string(str): The raw input string to be BPE't
        num_merges (int): The number of merge operations to perform.
        frac(float): The fraction of total tokens that occur in this specific touple

    Returns:
        tuple: A tuple containing:
            - set: The final vocabulary of tokens.
            - list: The learned merge rules in order.
            - set: The starting vocab.
    """
    word_counts = tokenization_dict(string)
    final_vocab, merge_rules,vocabold = bpe(word_counts, num_merges, frac)
    return final_vocab, merge_rules,vocabold

def tokencounter(text):
    words = text.lower().split()
    #tokenize leaving an artifact list
    textlist = [list(word+" ") for word in words]
    #removing artifact list
    textlist = [x for xs in textlist for x in xs]
    textlist = [w.replace(' ', '</w>') for w in textlist]
    
    return len(textlist)

if __name__ == "__main__":
    # Example usage of the full tokenization_dict and BPE pipeline
    newtext = False
    tl= False
    #sample_text = "House house house cat sat rat hand harry handicap andasda"
    f = open("shakes.txt")
    sample_text = f.read()
    f.close()
    if(newtext):
        final_vocab, merge_rules,vocabold = preprocessing(sample_text,200)
        with open("vocab.pkl", "wb") as fp:
            pickle.dump([final_vocab,merge_rules],fp)
        fp.close()
        merge_rules = []

    if tl:
        with open("vocab.pkl", "rb") as fp:
            [final_vocab,merge_rules,vocabold]=pickle.load(fp)
            fp.close()
    
        tl = tokenization_list(sample_text,merge_rules)

        with open("tl.pkl", "wb") as fp:
            pickle.dump(tl,fp)
        fp.close()
    else:
        with open("tl.pkl", "rb") as fp:
            tl=pickle.load(fp)
    print(tl[0:100])
    #print("compression rate "+ str(tokencounter(sample_text)/len(tl)))