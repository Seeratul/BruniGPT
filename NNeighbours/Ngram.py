import collections
import json
from typing import List, Tuple, Dict

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
           Must be 2 or greater.
        tokens: A list of strings (words) from which to build the model.

    Returns:
        A dictionary representing the n-gram model. Returns an empty dictionary
        if not enough tokens are provided to create a single n-gram.
    """
    if n < 2:
        raise ValueError("n must be at least 2 to have a preceding word context.")

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
        #print(context)

    for i in (ngram_counts):
        total = 0
        for j in (ngram_counts[i]):
            total += (ngram_counts[i][j])+1
        for k in (ngram_counts[i]):
            ngram_counts[i][k] = (ngram_counts[i][k]+1)/total

    return ngram_counts

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
  
