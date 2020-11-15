import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    result = {}
    for file in os.listdir(directory):
        cur_file = open(os.path.join(directory, file), "r", encoding="utf8")
        result[file] = cur_file.read()
        cur_file.close()
    return result


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    result = []
    # nltk.download('stopwords')
    preresult = nltk.tokenize.word_tokenize(document)
    for i in range(len(preresult)):
        # lowercase
        preresult[i] = preresult[i].lower()
        # not a stopword
        if preresult[i] not in nltk.corpus.stopwords.words("english"):
            to_add = ""
            # filter out punctuation
            for c in preresult[i]:
                if c not in string.punctuation:
                    to_add += c
            result.append(to_add)

    return result


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    result = {}
    n_docs = len(documents)
    # for every document
    for cur_doc in documents:
        # for every word in the current document
        for word in documents[cur_doc]:
            docs_with_word = 0
            if word not in result:
                # count documents with this word if we have not already
                for search_doc in documents:
                    if word in documents[search_doc]:
                        docs_with_word += 1
                result[word] = math.log(n_docs / docs_with_word)

    return result


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    result = []
    file_score = {}
    for file in files:
        # counting the score of each file
        score = 0
        # counting tf of every query word
        for word in query:
            tf = 0
            if word in files[file]:
                for search_word in files[file]:
                    if search_word == word:
                        tf += 1
            # summing the tf-idf of every query word
            score += tf * idfs[word]
        file_score[file] = score

    # finding the n top files
    for _ in range(n):
        max_file = ""
        for file in file_score:
            if max_file == "" or (file_score[file] > file_score[max_file] and file not in result):
                max_file = file
        result.append(max_file)
    return result


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    result = []
    sentence_score = {}
    sentence_den = {}

    for sentence in sentences:
        # counting the score of each sentence
        score = 0
        den = 0
        # counting tf of every query word
        for word in query:
            if word in sentences[sentence]:
                # summing the idf of every query word
                score += idfs[word]
                # calculating density
                for search_word in sentences[sentence]:
                    if search_word == word:
                        den += 1

        sentence_score[sentence] = score
        sentence_den[sentence] = den / len(sentences[sentence])

    # finding the n top sentences
    for _ in range(n):
        max_sentence = ""
        for sentence in sentence_score:
            if max_sentence == "" or (sentence_score[sentence] > sentence_score[max_sentence] and sentence not in result) or (sentence_score[sentence] == sentence_score[max_sentence] and sentence_den[sentence] > sentence_den[max_sentence] and sentence not in result):
                max_sentence = sentence
        result.append(max_sentence)

    return result


if __name__ == "__main__":
    main()
