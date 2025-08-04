Task 1 
(for code checkout main1.py)
All the measures for task 1 were eyballed as proper hpo was supposed to wait for Task 2+.

To get a good first look at the data I ran:
"tr 'A-Z' 'a-z' < shakes.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -nr | head"
in the console.
After which i performed the following steps:
- Perform Train and test split.
I used the provided file clean_nltk_shakespear_data.py to create a train test and eval set.
- Train the segmenter with varying k
I used compression as a sucess metric, diminishing returns were reached at 2000-4000k with a compression of ~1.6 to ~1.7. 
- Compare the performace against a different set
The compression was similar betwenn the training data and the "Attention is all you need" paper as test data 
with the test data being around 0.1 worse.
- Measure for accuracy.
I choose the delta between the test and the training data coompression as my accuracy metric. 
I made the decision that massively increasing k for minimal returns (above 4000) was not worth it.

Bonus:
As BPE preogressively merges subwords some subwords might end up "orphaned"/rarely used as a large chunk of their occurence might get swallowed by a bigger subword. 
Given that we want as much information as possible packed into as few and as general subwords as possible this is not ideal.
The solution to this was to run a second optimization/bonus round in which subwordes that have less occurence then the most common pair
 get unmerged and that most common pair gets turned into a subword instead. 
 The performance impact of this method will be discussed in Task 2. 
 In task one it improves compression from 1.6333 on test to 1.6335 using a k of 2000.
Bonus Bonus: For data symetry reasons i dont understand the optimization/bonus round always leads to and optimal 
set after k-1 passes. I think this is really neet and probably something already known, somwhere.

Task 2
(for code checkout main2.py)
- N gram engine.
Arbitrarily scalable engine that uses the highest 3 grams that have a match for the key with a 0.6,0.3,0.1 weighting respectively. 
Automaticly cuts text to size of highest gram and can default to using only 2 (0.7,0.3)grams or the unigram model. 
Hardware limited on my computer to n<=10grams.
Applies Laplace smoothing on generation.
- Ngramhpo 
Uses Ngram engine to calculate a matrix of perplexity relative to k/n for an engine generated with train.
k\n  1      2     3     4     5      6      7       8       9       10     
0    19.54  12.69 8.01  5.40  4.43   4.87   9.19    27.73   75.73   161.98
555  62.02  21.98 10.65 7.87  13.25  51.62  288.70  535.98  747.59  811.02
1111 87.67  24.66 11.17 9.38  34.57  136.58 563.48  840.79  1050.52 1102.25
1666 107.89 25.55 11.66 10.74 60.96  216.22 749.10  1066.36 1265.96 1317.31
2222 120.36 25.90 12.01 11.80 83.29  273.89 865.30  1221.01 1401.18 1460.29
2777 129.61 26.42 12.50 12.93 103.65 325.23 957.11  1348.57 1507.90 1575.73
3333 137.17 26.55 12.83 13.75 118.17 362.55 1014.33 1435.78 1598.99 1665.53
3888 143.07 26.65 13.13 14.45 129.79 387.91 1047.88 1481.65 1630.17 1692.87
4444 148.71 26.68 13.37 15.04 137.92 414.44 1091.95 1557.07 1707.65 1768.11
5000 151.81 26.68 13.56 15.56 144.21 430.47 1113.87 1584.74 1738.08 1797.25

- Extrinsic evaluation/sentencegen. 
Takes a touple containing an arbitrary number of strings (1+), an ngram model to use, 
the number of tokens to maximaly generate, the top results to sample from (1 for deterministic),
and utilizes end-of-sequence tokens(. ? !) for an early stop.