import sys
import numpy as np
from typing import Tuple

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"

import torch
from model import Encoder, Decoder, Seq2Seq
from utils import tokenize
from tqdm import tqdm


from evaluate import load_model, translate_sentence


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
class Evaluation:
    def __init__(self, device, checkpoint_path):
        self.device = device
        print('Loading Checkpoint ...')
        checkpoint = torch.load(checkpoint_path)
        print('Loading Model...')
        self.model = load_model(checkpoint, device)
        print('Loading Vocabulary...')
        self.source_stoi = checkpoint['source_stoi']
        self.source_itos = checkpoint['source_itos']
        self.target_stoi = checkpoint['target_stoi']
        self.target_itos = checkpoint['target_itos']
    
                 
    def predict(self, factor: str):

        expansion = translate_sentence(factor, self.source_stoi, self.source_itos, self.target_stoi,
                                               self.target_itos, self.model, self.device, max_len = 30)
        return expansion


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = 'checkpoints/checkpoint.pt'
    
    evaluation = Evaluation(device, checkpoint_path)
    print("Predicting Factors ....")
    

    factors, expansions = load_file(filepath)
    pred = [evaluation.predict(f) for f in tqdm(factors, total = len(factors))]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(f'Score: {np.mean(scores)}')
    

if __name__ == "__main__":
    
    
    
    main("test.txt" if "-t" in sys.argv else "train.txt")