Task 1 
All the measures for task 1 were eyballed as proper hpo was supposed to wait for Task 2+.

To get a good first look at the data I ran:
"tr 'A-Z' 'a-z' < shakes.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -nr | head"
in the console.
After which i performed the following steps:
- Perform Train and test split.
I used the provided file clean_nltk_shakespear_data.py to create a train test and eval set.
- Train the segmenter with varying k
I used compression as a sucess metric, diminishing returns were reached at 2000-4000k with a compression of ~2.0. 
- Compare the performace against a different set
The compression was similar betwenn the training data and the "Attention is all you need" paper as test data with the test data being around 0.1 worse.
- Measure for accuracy.
I choose the delta between the test and the training data coompression as my accuracy metric. I made the decision that massively increasing k for minimal returns (above 4000) was not worth it.

