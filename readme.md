Task 1 
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
The compression was similar betwenn the training data and the "Attention is all you need" paper as test data with the test data being around 0.1 worse.
- Measure for accuracy.
I choose the delta between the test and the training data coompression as my accuracy metric. I made the decision that massively increasing k for minimal returns (above 4000) was not worth it.

Bonus:
As BPE preogressively merges subwords some subwords might end up "orphaned"/rarely used as a large chunk of their occurence might get swallowed by a bigger subword. Given that we want as much information as possible packed into as few and as general subwords as possible this is not ideal.
The solution to this was to run a second optimization/bonus round in which subwordes that have less occurence then the most common pair get unmerged and that most common pair gets turned into a subword instead. The performance impact of this method will be discussed in Task 2. In task one it improves compression from 1.6333 on test to 1.6335 using a k of 2000.
Bonus Bonus: For data symetry reasons i dont understand the optimization/bonus round always leads to and optimal set after k-1 passes. I think this is really neet and probably something already known, somwhere.