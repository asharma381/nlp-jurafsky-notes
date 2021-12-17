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

![lecture1](/Users/aditya/Documents/nlp-notes/images/lecture1.png)



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



Case Folding:

* There's a difference between U.S. and us

