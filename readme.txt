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
I used compression as a sucess metric, diminishing returns were reached at 2000-4000k with a compression of ~2.4 to ~2.5 
I would also like to mention that i put a hard limit of 5 on the maximum segment lenght to help against overfitting onto long words.
- Compare the performace against a different set
The compression was similar betwenn the training data and test data with an difference of 0.03
- Measure for accuracy.
I choose the delta between the test and the training data coompression as my accuracy metric. 
I made the decision that massively increasing k for minimal returns (above 4000) was not worth it.

Bonus:
As BPE preogressively merges subwords some subwords might end up "orphaned"/rarely used as a large chunk of their occurence might get swallowed by a bigger subword. 
Given that we want as much information as possible packed into as few and as general subwords as possible this is not ideal.
The solution to this was to run a second optimization/bonus round in which subwordes that have less occurence then the most common pair
get unmerged and that most common pair gets turned into a subword instead. 
The performance impact of this method will be discussed in Task 2. 
In task one with k 2000 it improves compression on the training set by 0.002 with the test set being unaffected.
This could be due to the small size of the dataset or because this postrocessing is frankly a waste of time.
Given extremely small ks like 50 there is an improvement for the test data of 0.006 which is an iprovement of compression by 5%.
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
k\n  1      2     3     4      5      6       7       8       9       10
0    19.54  12.69 8.01  5.40   4.43   4.87    9.19    27.73   75.73   161.98
500  102.77 40.00 19.89 22.06  89.79  355.87  673.02  831.37  872.47  894.87
1000 179.34 52.28 26.41 57.55  288.82 830.61  1118.71 1212.15 1230.16 1236.98
1500 228.31 57.39 30.46 95.49  405.07 1028.83 1301.42 1382.44 1401.22 1404.65
2000 262.22 59.61 33.28 121.54 483.39 1132.17 1381.69 1477.06 1495.08 1496.36
2500 283.43 61.21 35.15 138.84 523.62 1195.62 1435.96 1533.37 1550.86 1550.97
3000 302.60 63.20 37.30 158.55 586.00 1289.84 1524.02 1617.60 1629.57 1631.60
3500 314.57 63.86 38.80 172.17 639.30 1386.91 1625.36 1713.39 1724.29 1726.69
4000 316.41 63.84 39.44 170.36 623.68 1363.00 1624.26 1705.95 1725.54 1729.40
4500 319.57 63.85 39.91 173.12 630.66 1410.81 1689.50 1769.47 1793.44 1797.40

- Extrinsic evaluation/sentencegen. 
Takes a touple containing an arbitrary number of strings (1+), an ngram model to use, 
the number of tokens to maximaly generate, the top results to sample from (1 for deterministic),
and utilizes end-of-sequence tokens(. ? !) for an early stop. 
Sadly the gernation is pure gibberish for the models with lowest perplexity.
It would certainly be possible to provide nicer generation that is overfitted.


Task 3
***GEN IS BROKEN***
(main3)
Task 3 was programmed entirely without pytorch with the exception of Legacy batching code.
(Time constraints)
The following components were hand made:
- A one hot encoder that turns the input into a one hot vector
- The embedding table
- The entire forward pass
- The entire backward pass including weight updates using sgd with batchsize 1 blocksize 1
- A rudimentary version of the above that uses batches of larger sizes to speed up processing
  utilizing np´s efficent matrix multiplications.
- Proper SGD 
- A generate function to utilize the model
- Saving and loading
- Calculating perplexity

I probably spent way to much time on this but i really enjoyed trying to work it all out.
Hihlights include problems with the learning rate, efficent matrix multiplication of 3d matirces and multiple hours
of (futilly) attempting to find out how i ended up doing ascent instead of descent. 
My implementation currently struggles with learning larger k`s due to hardware limitations
but performs adequately for k=1. While I would love to spend more time on this I think
this is a respectable result given the time and i should really move on to task4.
In the demo with k=1 and a rather short runtime a perplexity of 26.13 
on the validation set was achieved.
Below Pictures.

**Pictures**

Task 4
(main4)
Implementation:
Its a standard GPT implementation along the lines of the provided examples.
It contains:
- A self build transformer block using GELU
- An optimized data loader for batching
- The Gelu activation function
- A one call hook to run it for HPO with certain parameters
- A visualization tool
- A tool for ther generated output

