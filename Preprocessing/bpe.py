import pickle
import collections 

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
    merge_dict = {}
    #split and set lower case
    words = text.lower().split()
    #tokenize leaving an artifact list
    textlist = [list(word+" ") for word in words]
    #removing artifact list
    textlist = [x for xs in textlist for x in xs]
    textlist = [w.replace(' ', '</w>') for w in textlist]
    for i in range(len(merge_rules)):
        if  merge_rules[i][0] in merge_dict:
            merge_dict[merge_rules[i][0]].append(merge_rules[i][1])
        else:
            merge_dict[merge_rules[i][0]] = [merge_rules[i][1]]
        print(merge_dict[merge_rules[i][0]][0])
        print(len(merge_dict[merge_rules[i][0]][0]))


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

def unmerge(pair,v_in):
    """Merge the most frequent pair in the vocabulary keys."""
    v_out = {}
    bigram = " "+' '.join(pair) +" "
    replacement = " "+''.join(pair)+ " "
    for word, freq in v_in.items():
        # Replace the pair in the word string
        new_word = word.replace(replacement, bigram)
        v_out[new_word] = freq
    return v_out

def bpe(word_counts, num_merges,extra_runtime=0,frac = 0):
    """
    Performs Byte-Pair Encoding to learn merge rules and create a vocabulary.

    Args:
        word_counts (dict): A dictionary of words and their frequencies.
        num_merges (int): The number of merge operations to perform.
        extra_runtime(int): how often it tries to improve the tokens
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
    startingvocab = set()
    vocabold = vocab
    for word in vocabold.keys():
        startingvocab.update(word.split(' '))
    merges = []

    for i in range(num_merges):
        # 2. Get statistics of adjacent pairs
        stats, symboln = get_stats(vocab)
        if not stats:
            raise ValueError('Your merge number leads back to the inital dataset.')
        # 3. Find the most frequent pair
        best_pair = max(stats, key=stats.get)
        #4. merge it
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)

    for i in range(extra_runtime):
        stats, symboln = get_stats(vocab)
        best_pair = max(stats, key=stats.get)
        mv = stats[best_pair]
        mini = 0
        for i in range(len(merges)):
            key =''.join(merges[i])
            merge = merges[i]
            if key in symboln:
                value = symboln[key]
            else:
                value = 0
            if value < mv:
                mv = value
                mini = i
        if mv < stats[best_pair]:
            vocab = unmerge(merges[mini],vocab)
            merges.pop(mini)
            stats, symboln = get_stats(vocab)
            if not stats:
                raise ValueError('Your merge number leads back to the inital dataset.')
            # 3. Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            #4. merge it
            merges.append(best_pair)
            vocab = merge_vocab(best_pair, vocab)

        else:
            print("Tokens Fully Optimized")
            break

            #print(merges)
            #print(symboln[key])
       #if (stats[max(stats, key=stats.get)]>):
           
        #if stats[max(stats, key=stats.get)]:
            #return 0
    
     
    print("All merges performed")

    # 5. Create the final token vocabulary from the keys of the learned vocab
    final_tokens = set()
    for word in vocab.keys():
        final_tokens.update(word.split(' '))


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
    sample_text = "House house house cat sat rat hand harry handicap andasda"
    #f = open("shakes.txt")
    #sample_text = f.read()
    #f.close()
    runtime = 10
   
    final_vocab, merge_rules,vocabold = preprocessing(sample_text,5,runtime)
    tl = tokenization_list(sample_text,merge_rules)
    print("compression rate "+ str(tokencounter(sample_text)/len(tl)))

 