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
  * Choose a random bigram according to its probability $P(<s>,w)$, string together 
* Approximate Shakespeare using unigrams, bigrams, trigrams,  quadgrams.
* The perils of overfitting: n-grams only work well if the test corpus looks like the training corpus
  * In real life this doesn’t often occur, need to generalize 

* One kind of generalization **Zeros** 
  * Bigrams with 0 probability, means thar we will assign 0 probability to the test set!
