import re
import string
from pathlib import Path
import argparse

import nltk
from nltk.corpus import wordnet


CORPUS = nltk.corpus.gutenberg
# CORPUS = nltk.corpus.words


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

    # Matches strings containing only digits/punctuations(with leading, trailing spaces)
    num_punct_re = re.compile(fr'\s*[0-9{re.escape(string.punctuation)}]+\s*$')

    for word, pos in words:
        if num_punct_re.match(word):
            continue
        clean_words.append((word, pos))
        
    return clean_words


def main(transcript_file_path: Path, output_file_path: Path):
    '''Write words to output file which are not in the corpus.
    Compare lemmatized words also to get better accuracy'''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    dict_words = set([w.lower() for w in CORPUS.words()])
    text = transcript_file_path.read_text(encoding='utf-8')

    non_english_words = set()

    for sentence in nltk.tokenize.sent_tokenize(text):
        words = nltk.tokenize.word_tokenize(sentence)
        words_with_pos = nltk.pos_tag(words)
        clean_words_with_pos = get_clean_words(words_with_pos)
        for word, pos in clean_words_with_pos:
            wordnet_pos = get_wordnet_pos(pos)
            lemma = lemmatizer.lemmatize(word, wordnet_pos)
            if word.lower() not in dict_words and lemma.lower() not in dict_words:
                non_english_words.add(lemma)

    output_file_path.parent.mkdir(exist_ok=True)
    output_file_path.write_text('\n'.join(sorted(non_english_words)))


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'transcript_path',
        type=Path,
        help='Path to transcript file'
    )
    arg_parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('./output.txt'),
        help='Path to output file. Default: ./output.txt'
    )
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        transcript_file_path=args.transcript_path,
        output_file_path=args.output
    )