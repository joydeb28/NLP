import embeddings
import numpy as np
import argparse


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_parser():
    parser = argparse.ArgumentParser(
        prog='ELMO vs NNLM',
        usage='python comparison_test.py [sentence1] [sentence2]',
        description='This module demonstrates comparison of ELMo and NNLM',
        add_help=True
    )

    parser.add_argument("sentences", nargs=2, help="The sentences to compare cosine similarity")

    return parser


if __name__ == "__main__":
    parser = get_parser().parse_args()
    sentence1, sentence2 = parser.sentences

    # Get embeddings corresponding to each sentences
    results_elmo, results_nnlm = embeddings.get_embeddings_elmo_nnlm([sentence1, sentence2])

    print("[Cosine Similarity]")
    print("\"{}\" vs \"{}\"".format(sentence1, sentence2))
    print("ELMo:", cos_sim(results_elmo[0], results_elmo[1]))
    print("NNLM:", cos_sim(results_nnlm[0], results_nnlm[1]))
