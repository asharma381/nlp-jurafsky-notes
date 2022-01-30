<h1>NLP || Dan Jurafsky || Stanford University</h1>



<h4>Lecture 1 - NLP Course Information</h4>

Applications:

* Question-Answering: IBM's Watson (Won Jeopardy on Feb 16, 2011)
* Information Extraction:  Create calendar entry from given email
* Sentiment Analysis: Reading reviews about cameras, find attributes (affordabilty, size, weight)
  * Ex: {Nice and compact to carry, light and small, feels flimsy is plastic and delicate}
* Machine Translation: Fully automatic (entire phrase from Chinese to English) or Helping Humans translate (possible next words)

<center>Language Technology</center>

| Mostly Solved             | Making good progress     | Still really hard       |
| ------------------------- | ------------------------ | ----------------------- |
| Spam Detection            | Sentiment Analysis       | Question Answering (QA) |
| Parts-of-speech tagging   | Coreference Resolution   | Paraphrase              |
| Named entitry recognition | Word sense disambiguirty | Summarization & Dialog  |

![lecture1](https://github.com/asharma381/nlp-jurafsky-notes/blob/master/images/lecture1.png)



Ambiguity makes NLP hard

Tools:

- Knowledge about language
- Knowledge about the wrold
- A way to combine the knowledge sources

How do we generalize this:

* Probabilitistic models built form language data

Covers key theory and models for statsitical NLP:

* Viterbi
* Naïve Bayes, Maxent Classifiers
* N-gram Language Models
* Statistical Parsing
* Inverted index, tf-idf, vector models of meaning



Robust real-world applications:

* Information extraction
* Spelling Correction
* Information Retrieval



Pre-reqs: Simple linear algebra (vectors, matrices), Basic probabilty theory, Python Programming





<h4>Lecture 2 - Basic Text Processing</h4>

Regular Expression is the most basic and fundamental tool for text processing



Disjunctions:

* Letters inside square brackets: 

​		[wW]oodchucks = {woodchucks, Woodchucks}

​		[0123456789] = Any digit

* Ranges:
  [A-Z] = Any uppercase letter = Ex: {<b>D</b>renched blossoms}
  [a-z] = Any lowercase letter = Ex: {<b>m</b>y beans are so important}
  [0-9] = A single digit = Ex: {Chapter <b>1</b>:}

Regexpal = JavaScript tool for regular expression



* Negations:
  [ ^ A-x] - Carat means negation only when inside []
* Pipe | = means [A | B], either A or B
* Optional Previous Chars: [colou?r]
* Kleene Operators (* and +) - Stephen C Kleene
  * 0 or more previous chars: [oo*h!] = {oh!, ooh!, ooooh!}
  * 1 or more previous chars: [o+h!] =  {oh!, ooh!, ooooh!}

* [beg.n] = {begin, begun, began}
* Anchors (^ and $)
  * ^[A-Z] = matches the 1st instance of capital letter found
  * [A-Z]$ = matches the last instance of capital letter found

* Period = `\.` (backslash + ''."), since period matches everything
* Goal: Find all `the's `
  * `[^A-Za-z][Tt]he [^A-Za-z]`

* Errors:
  * Type I Errors (False Positives): Matching strings that shouldn't have been matched 
    (ex: "there" matched when "the")
  * Type II Errors (False Negatives): Matching strings that should have been matched but weren't 
    (ex: "The") 

* Increasing the Accuracy and Precision $\rightarrow$ Minimizes False Positives
* Increasing the Coverage or Recall $\rightarrow$ Minimizes False Negatives



<h4>Lecture 3 - Regular Expressions in Practical NLP</h4>

Stanford English Tokenizer - deterministic, fast-high quality tokenizer

* Abbreviations:
  * ABMON = Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec
  * ABDAY = Mon | Tue | Wed | Thu | Fri | Sat | Sun



<h4>Lecture 4 - Word Tokenization</h4>

Every NLP Task needs to be able to do text normalization:

1. Segmenting/Tokenizing Words into runnings text
2. Normalize Word Formats
3. Segmenting sentences in running text

**Lemma**: same stem, part of speech, rough word sense, belongs to the same root (ex: cat, cats)

**Wordform**: the full inflected surface form (cat and cats are different wordforms)

Ex:  ``they lay back in the San Francisco grass and looked and the stars and their``

* **Type**: an element in the vocabulary. 
  * $V = \text{vocabulary} = \text{set of types}$ 
  * $|V| \text{ is the size of the vocabulary}$
* **Token**: an instance of that type in running text.     
  * $N = \text{number of tokens}$

In example, there are 15 tokens and 13 types (words: and, the repeat).

UNIX Text Processing, Given shakes.txt file of Shakespeare's work

`tr 'A-Z' 'a-z'< shakes.txt | tr -sc 'A-Za-z' '\n' < shakes.txt | sort | uniq -c | sort -n -r | less` $\rightarrow$  Treats lowercase and uppercase the same, replaces all spaces with new line, sorts the words, finds unique words, and sorts then with order of most frequently appearing

 Issues with Tokenization:

* Finland's => Finland, Finlands, Finland's
* what're => what are
* State-of-the-art => state of the art
* lowercase => lower-case, lowercase, lower case
* PhD. => PhD ,PhD. Dr.

<h4>Lecture 5 - Word Normalization and Stemming</h4>

Need to "normalize" terms: Information Retrieval $\rightarrow$ indexed terms and query terms must have same form

Case Folding:

* There's a difference between U.S. and us
* Reduce all letters to lower case, exceptions: Fed vs fed, SAIL vs. sail

Lemmatization: reduce inflection or variant forms to base form: am, are, is $\rightarrow$ be

* Machine Translation (Spanish): Quiero (I want), Quieres (you want), same lemma as **querer** 'want'

Morphology: morphemes

* The small meanigful units that make up words

* Stems: The core meaning-bearning units

  * Task: Reduce terms to their stems in information retrieval

  * Automat(a), Automatic, Automation all reduced to **automat**

  * Porter's Algorithms: The most common English stemmer:

    * ies $\rightarrow$ i
    * (v)ing $\rightarrow \empty$ walking to walk 

    * `tr -sc 'A-Za-z' '\n' < shakes.txt | tr 'A-Z' 'a-z' | grep 'ing$' | sort | uniq -c | sort -n -r | less` words ending in 'ing' in Shakespeare.
    * For example: {nothing, something, king, sing} shouldn't be remove

* Affixes: Bits and Pieces that adhere to stems

  * Often with grammatical functions

<h4>Lecture 6 - Sentence Segmentation and Decision Trees</h4>

* ! and ? are relatives unambiguous
* Period '.' is quite ambiguous sentence boundary
  * Ex: Inc. or Dr. or 2.3
  * To solve this problem: Build a binary classifier
    * Looks at '.' to decide of EndOfSentence/notEndOfSentence
    * Classifiers: hand-written rules, regular expressions, machine-learning
* Decision Trees: Is an if-then-else statement
  * Choosing Features is important part
  * Setting up the structure is too hard to do by hand (numeric features, pick threshold)
  * Features can be exploited by any kind of classifier: {Logisitic Regression, SVM, Neural Nets}
* Look at the words before or after a period
* Length of the word, probability that ends a sentence or starts a sentence. (ex: `. The` is start of sentence)

<h4>Lecture 7 - Minimum Edit Distance</h4>

String Similarity

* Spell Correction: `graffe`
* Computational Biology (Assign two sequences of nucleotides)
* Machine Translation, Information Extraction, Speech Recognition



Minimum Edit Distance: minimum number of editing operations

* Insertion
* Deletion
* Substitution

<img src="https://github.com/asharma381/nlp-jurafsky-notes/blob/master/images/lecture7.png" alt="lecture1" style="zoom: 50%;" />

Each Operation has a cost of 1 (Distance between them is 5)

Levenshtein Distance (cost for substitution is 2) Total Cost = 8



Evaluating Machine Translation - how well a machine translation system does

* Measure the words inserted, deleted, and substtitioed by comparing two sentences.

Named Entitity Extraction: improve accuracy

* Stanford President .... vs Stanford University President = are the same, we can use same approach
* IBM vs IBM Inc.
  

Define Min Edit Distance:

* Given two strings $X$ of length $n$ and $Y$ of length $m$. Define $D(i,j)$ as distance matrix 



<h4>Lecture 8 - Computing Minimum Edit Distance</h4>

Dynamic Programming: A tabular method of computation for $D(n,m)$. Solving problems by combining solutions to subproblems

Bottom-up: Compute $$D(i,j)$$ for small $i,j$. Compute larger $D(i,j)$ based on prevoiusly computed smaller values



Formal Levenshtien Distance Calculation:

* Initialization
  $D(i,0) = i$
  $D(j,0) = j$

*  Recurrence Relation

  $D(i,j) = min \begin{cases} D(i-1,j) + 1 \\ D(i,j-1) + 1 \\ D(i-1,j-1) + \begin{cases} 2; \text{ if } X(i) \neq Y(j) \\ 0; \text{ if } X(i) = Y(j) \end{cases}\end{cases}$

* Termination: $D(N,M)$ is distance



Levenshtien Distance Dynamic Programming Table

|       | #     | E     | X     | E     | C     | U     | T     | I     | O     | N     |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **#** | **0** | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     |
| **I** | **1** | 2     | 3     | 4     | 5     | 6     | 7     | 6     | 7     | 8     |
| **N** | 2     | **3** | 4     | 5     | 6     | 7     | 8     | 7     | 8     | 7     |
| **T** | 3     | 4     | **5** | 6     | 7     | 8     | 7     | 8     | 9     | 8     |
| **E** | 4     | 3     | 4     | **5** | **6** | 7     | 8     | 9     | 10    | 9     |
| **N** | 5     | 4     | 5     | 6     | 7     | **8** | 9     | 10    | 11    | 10    |
| **T** | 6     | 5     | 6     | 7     | 8     | 9     | **8** | 9     | 10    | 11    |
| **I** | 7     | 6     | 7     | 8     | 9     | 10    | 9     | **8** | 9     | 10    |
| **O** | 8     | 7     | 8     | 9     | 10    | 11    | 10    | 9     | **8** | 9     |
| **N** | 9     | 8     | 9     | 10    | 12    | 12    | 11    | 10    | 9     | **8** |



<h4>Lecture 9 - Backtrace for Computing Alignments</h4>

Need to **align** each character of the two strings to each other

Backtrace - every time we enter a cell, remember where we came from, then trace back the path

$D(i,j)$ = {insertion (left), deletion (right), substitution (diagonal)} in the Distance Matrix

Time Performance $\mathbb{O}(nm)$, Space $\mathbb{O}(nm)$, Backtrace $\mathbb{O}(n+m)$

<h4>Lecture 10 - Weighted Minimum Edit Distance</h4>

Add weights to the computation:

* Spell Correction: some letters are more likely to be mistyped than others (vowels more confused)
* Biology: certain kinds of insertion and deletion are more likely to occur

Confusion Matrix can see which two letters are more likely to be missed (ex: a and e, keyboard distance)

Change Formula: Add the cost for adding and deleting for each character

<h4>Lecture 11 - Minimum Edit Distance for Computational Biology</h4>

* Comparing genes or regions from different species, assembling fragments of DNA
* In NLP, we can talk about distance (minimized) and weights

Needleman-Wunsch Algorithm: Positive Cost for matching, negative cost for deletions



<h4>Lecture 12 - Introduction to n-grams</h4>

Probabilistic Language Modeling: assign a probability to a sentence

* Machine Translation: P(**high** winds tonite) > P(**large** winds tonite)
* Spell Correction: P(about fifteen **minutes**) > P(about fifteen **minuets**)
* Speech Reconition: P(I saw a van) >> P(eyes awe of an)
* Summarization, Question Answering

Given a sequence of words $P(W) = P(w_1, w_2, w_3, \ldots, w_n)$

Find the probability of word $P(w_5 | w_1, w_2, w_3, w_4)$

**Language Model (LM)**: a model that computes the probability of $P(W)$ or $P(w_2 | w_1)$. 

Chain rule of probability: Conditional Probability $P(A|B) =  P(A \cap B)/ P(B)$ 

Simplest Model: Unigram model (predicts just based on 1 both)

Bigram: find the probability based on the previous word

Extend onto trigrams, 4-grams, n-grams (language has long-distance dependencies)

<h4>Lecture 13 - Estimating n-gram probabilites</h4>

Bigrams Example from Resurant Food

$P(want | spend) = 0$ 	Here, $P = 0$ is caused by a grammatical error. {spend want} is incorrect

$P(food | to) = 0$ 			Here, $P = 0$ is caused by a contingent error. {to food} never present in data



Practically, we do everything in log space (arithmetic underflow, faster to add)

$p_1 * p_2 * p_3 * p_4 = \log p_1 + \log p_2 + \log p_3 + \log p_4$

 Google n-gram Corpus - 1 trillion words, 1 five-word sequence, 13 million unique words

## Lecture 14 - Evaluation and Perplexity

* Train parameters of our language model on a **training set**
* Test the model's performance on data we haven't seen before
  * **Test Set**: unseen dataset that is different from our training set, totally unused.
  * **Evaluation Metric** tells us how well our model does on the test set
* Extrinsic Evaluation: Put 2 models in a task, run the task and get an accuracy. Compare the two
  * Difficulties: Time consuming, 
* Intrinsic Evaluation: Perplexity
  * Bad Approximation: Unless test data and training data are similar, only useful in pilot experiments

The Shannon Game: Intiuitiion of Perplexity

* How can we predict the next word
* Unigrams are terrible at this - only have 1 word to refer to 
* The best language model is one that best predicts an unseen test set
* **Perplexity**: probability of the test set, normalized by the number of words
  * Minimizing perplexity is the same as maximizing probability, Lower perplexity = better model
  * $PP(W) = P(w_1w_2\ldots w_n)^{-1/N}$

| N-gram Order | Unigram | Bigram | Trigram |
| ------------ | ------- | ------ | ------- |
| Perplexity   | 962     | 170    | 109     |

Training 38 million words, test 1.5 million words, on WSJ

## Lecture 15 - Generalization and Zeros

* Shannon Visualization Method
  * Choose a random bigram according to its probability `P( <s> ,w)`, string together 
* Approximate Shakespeare using unigrams, bigrams, trigrams,  quadgrams.
* The perils of overfitting: n-grams only work well if the test corpus looks like the training corpus
  * In real life this doesn’t often occur, need to generalize 

* One kind of generalization **Zeros** 
  * Bigrams with 0 probability, means thar we will assign 0 probability to the test set!

## Lecture 16 - Smoothing Add One

* Intuition of Smoothing: steal probability mass distribution to generalize better
* **Add-one Estimation** (Laplace Smoothing): pretend we saw each word one more time than we did
  * Just add one to all the counts
  * MLE Estimate: $P_{MLE}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
  *  Add-one Estimate: $P_{Add-1} = \frac{c(w_{i-1}, w_i) + 1}{c(w_{i-1}) + V}$
* Maximum Likelihood Estimate
  * of some parameter of a model $M$ from a training set $T$
  * maximizes the likelihood of the training set $T$ given the model $M$
* Suppose the word "bagel" appears 400 times in a million size corpus. MLE estimation = 400/1,000,000
* Example: Laplacian smoothed bi-gram counts (add-one to all the examples with 0 count)
  * Laplace-smoothed bi-grams reassesses the probability and reconstituted counts
* Add-one estimation is a very blunt instrument: oftentimes isn't used for N-grams
* However, it is used for other NLP models such as text classification or in domains where the number of 0's isn't huge

## Lecture 17 - Interpolation

* Sometimes it helps to use _less_ context
* **Backoff**: Use tri-grams if you have good evidence, otherwise bi-grams, then uni-grams
* **Interpolation**: Mix uni-grams, bi-grams, tri-grams
* In practice, interpolation works better.

**Linear Interpolation**: 
$\hat P(w_n | w_{n-1} w_{n-2}) = \lambda_1 P(w_n | w_{n-1} w_{n-2}) + \lambda_2 P(w_n | w_{n-1}) + \lambda_3P(w_n)$, where $\sum_i \lambda_i = 1$
* Lambdas Conditional on context can also be used
* Choose $\lambda$'s to maximize the probability of held-out data:
  * Use a held-out corpus {Training Data, Held-out Data, Test Data}
  * Fix the n-grams probabilities (on the training data)
  * Search for $\lambda$s that give largest probability to held-out set
* If we know all the words in advance: 
  * Vocabulary $V$ is fixed (closed vocabulary task)
* However, oftentimes:
  * Out of Vocabulary = OOV words (open vocabulary task)
  * Create an unknown word token `<UKN>`
  * Training of `<UKN>` probabilities
    * Create a fixed lexicon $L$ of size $V$
    * At text normalization phase, any training word not in $L$ changed at `<UKN>`
    * Now we train its probabilities like a training word
  * At decoding time
    * If text input: Use UKN probabilities for any word not in training
* Huge web-scale n-grams (Google N-gram corpus)
  * Pruning: Only store N-grams with count > threshold
  * Remove singletons of higher-order n-grams
* Efficiency
  * Efficient data structures like tries
  * Bloom Filters: approximate language models
  * Store words as indexes, not string
    * Use Huffman coding to fit large numbers instead of words into two bytes
  * Quantize probabilities (4-8 bits instead of 8-byte float)
* Smoothing for N-grams
  * Add-one smoothing (not for language modeling)
  * Commonly used method (extended interpolated Kneser-Ney)
  * large N-grams like Web: Stupid Backoff
* Discriminative models:
  * Choose n-gram weights to improve a task, not to fit the training set
* Parsing-based models
* Caching Models:
  * Recently used words are more likely to appear
  * Perform very poorly for speech recognition

## Lecture 18 - Good Turing Smoothing

* Advanced smoothing algorithms
  * Good-Turing
  * Kneser-Ney
  * Witten-Bell
  * Use the count of things they've seen **once** to help estimate the counts they have **never** seen
* Notation: $N_c = $ frequency of frequency $c$
  * $N_c$ is the count of things we've seen $c$ times
  * Ex: `Sam I am I am Sam I do not eat`
  * `I = 3, Sam = 2, am = 2, do = 1, not = 1, eat = 1`
  * $N_1 = 3, N_2 = 2, N_3 = 1$
* $P^*_{GT}$(things with zero frequency) = $\frac{N_1}{N}$, $c^* = \frac{(c+1)N_{c+1}}{N_c}$

## Lecture 19 -  Kneser Ney Smoothing 
* Better estimate of the probabilities of lower-order uni-grams
* $P_\text{continuation}(w)$ - for each word, count the number of bi-gram types it completes
  * Every bi-gram was a novel continuation
  * Normalized by the total number of word bi-gram types

$P_{KN} (w_i | w_{i-1})$ = $\frac{\text{max}(c(w_{i-1}, w_i) - d, 0)}{c(w_{i-1})} + \lambda(w_{i-1}) P_\text{continuation}(w_i)$
* $\lambda$ is a normalizing constant; the probability mass we've discounted
$\lambda(w_{i-1}) = \frac{d}{c(w_{i-1})} | \{w: c(w_{i-1}, w_i > 0)\}|$

## Lecture 20 -  Spell Correction Task

* Applications: Word Processing (suggestions), Web Search, Phones (Messages)
* Spell Tasks: Spell Error Detection
  * Spell Error Correction: Autocorrect, Suggest a correction, Suggestion lists
* Types of spelling errors
  * Non-word Errors (graffe $\rightarrow$ giraffe)
  * Real word Errors
    * Typographical Errors (three $\rightarrow$ there)
    * Cognitive Errors (homophones): (piece $\rightarrow$ peace, too $\rightarrow$ two)
* Non-word Spelling Error Detection: Any word not in a **dictionary** is an error, the larger the dictionary the better
* Non-word Spelling Correction: 
  * Generate candidates: real words that are similar to error
  * Choose the weighted edit distance, Highest noise channel probability

## Lecture 21 - The Noisy Channel Model of Spelling 
* Noisy Channel Intuition (original word $\rightarrow$ noise word)
  * To guess the word, take a set of hypothesis words decode them through the noisy channel, and find the noisy hypothesis word which looks most like the noisy word
  * $\hat w = \text{argmax}_{w \in V} P(w|x)$
    $\hat w = \text{argmax}_{w \in V} P(x|w)*P(w)$
  * Channel error model = likelihood * prior

* Misspelled word: `acress`
  * Find words with similar spelling (small edit distance to error)
  * Find words with similar pronunciation (small edit distance of pronunciation to error)
  * 80% of errors are within edit distance of 1
  * Also allows insertion of space or hyphen

* Create confusion matrix of spelling errors for substitution of $X$ for $Y$
  1. Use Noisy Channel
  2. Language Model (uni-gram or bi-gram)
* Multiply the two probabilities $k *P_1 * P_2$ by $k$ to normalize

## Lecture 22 - Real Word Spell Correction
* There may be instances where real words in the dictionary have been incorrectly used (ex: `minute` to `minuet`, `by` to `bye`)

* Given a sentence {$w_1, w_2, \ldots, w_n$}
* Generate a set of candidates for each word $w_i$
  * Candidate($w_1$) = $w_1, w_1', w_1''$
* Choose the sequence of candidates $W$ that maximizes $P(W)$
  * Get probabilities from Language Model {uni-gram, bi-gram}
  * Channel Model: same as for non-word spelling correction, plus need probability for no error, $P(w|w)$

## Lecture 23 - Spelling Correction and the Noisy Channel
* Spelling Correction
  * Very confident: Autocorrect
  * Less Confident: Give best correction, Give correction list
  * Not confident: flag as an error
* State of the Art Noisy Channel
  * $\hat w = \text{argmax}_{w \in V} P(x|y) P(w)^{\lambda}$
  * Weight the probability, learn $\lambda$ from a development test set
* Phonetic Error Model
  * Metaphone: convert misspelling to metaphone pronunciation
  * Nearby keys in classical keyboard
* Classifier-Based Methods
  * Use many features in a classifier for specific pairs
  * Ex: `weather, whether` - Check for 'cloudy' appearing within $\pm$10 words, verb after, __ or not

## Lecture 24 - What is Text Classification
* Spam Detection
  * All the features such as (subject line, links, urgency) can be combined in a classifier to give some evidence
* Tasks: Authorship Attribution, Male/Female Authorship, Positive/Negative Movie Review, Subject of Article via Categories
  * Solved using Bayesian Methods

* Text Classification: definition
  * Input: a document D, fixed set of classes C = {c1, c2, ..., cn}
  * Output: a predicted class c in C
* Hand-coded rules: Black-list-address, accuracy can be high, building and maintaining these rules is expensive

* Classification Methods: Supervised Machine Learning
  * Input: document, set of classes, set m hand-labeled documents (d1,c1), (d2,c2)
  * Output: a learned classifier y: d -> c
  * Kinds of classifiers: Naive Bayes, Logistic Regression, Support Vector Machines (SVMs), k-Nearest Neighbors

## Lecture 25 - Naive Bayes 
* Naive Bayes Intuition
  * Simple classification method based on Bayes Rule
  * Relies on very simple representation of documents (bag of words)
* The bag of words representation
  * Build a function gamma, which takes document D and returns a class (positive or negative)
  * Look at individual words in the document (subset of words) loses the order, take set of words and their counts
  * Ex: w1 occurs x1 times, w2 occurs x2 times.
  * Represent document by list of words and their counts


## Lecture 26 - Formalizing the Naive Bayes Classifier
* For a document D and a class C, P(c|d) = P(d|c) * P(c)/P(d)
* C_map = argmax P(c|d)
  * argmax P(d|c) * P(c)/P(d)  - Bayes Rule
  * argmax P(d|c) * P(c) - drop the denominator constant term P(d) is identical

* C_map = argmax P(x1, x2, ..., xn | c) * P(c) - count the relative frequencies in a corpus
* Bag of words assumption: Assume position doesn't matter (only care about features)
* Conditional Independence: Assume the feature probabilities P(xi|cj) are independent given the class c

Multinomial Naive Bayes Classifier
* P(x1, ..., xn | c) = P(x1 | c) * P(x2 | c) * P(x3 | c) * ... *  P(xn | c)
* C_nb = argmax P(c_j) PI_MULTI_ [P(x|c)]
* positional <-- all word positions in test document, assign the classes to the document
