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
* 0 or more previous chars: [oo*h!] = {oh!, ooh!, ooooh!}
* 1 or more previous chars: [o+h!] =  {oh!, ooh!, ooooh!}
* [beg.n] = {begin, begun, began}

