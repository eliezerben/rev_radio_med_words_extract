# 
# It's better to give a full meaningful sentence to pos_tagger because it uses the surrounding context to assign POS
# Lemmatizer is case sensitive
#
import re
import string
from pathlib import Path
import argparse

import nltk
from nltk.corpus import wordnet


# CORPUS = nltk.corpus.gutenberg
CORPUS = nltk.corpus.words


def get_wordnet_pos(treebank_tag):
    '''Convert treebank pos tag set to wordnet format'''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def get_clean_words(words):
    '''Remove words containing only numbers, punctuations'''
    clean_words = []

    # Matches strings containing only digits and punctuations(with leading, trailing spaces)
    num_punct_re = re.compile(fr'\s*[0-9{re.escape(string.punctuation)}]+\s*$')

    for word, pos in words:
        if num_punct_re.match(word):
            continue
        clean_words.append((word, pos))
        
    return clean_words


def main(transcript_file_path):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    dict_words = set([w.lower() for w in CORPUS.words()])
    text = transcript_file_path.read_text(encoding='utf-8')

    non_english_words = set()

    for sent in nltk.tokenize.sent_tokenize(text):
        sent_words = nltk.tokenize.word_tokenize(sent)
        words_with_pos = nltk.pos_tag(sent_words)
        clean_words_with_pos = get_clean_words(words_with_pos)
        for word, pos in clean_words_with_pos:
            wordnet_pos = get_wordnet_pos(pos)
            lemma = lemmatizer.lemmatize(word, wordnet_pos).lower()
            if word not in dict_words and lemma not in dict_words:
                non_english_words.add(lemma)


    Path('temp_words.txt').write_text('\n'.join(non_english_words))


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'transcript_path',
        type=Path,
        required=True,
        help='Path to transcript file'
    )
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.transcript_path)