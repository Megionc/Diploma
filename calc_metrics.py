from evaluate import load
import pathlib
import collections


import pandas as pd

import tqdm

import pathlib
import evaluate




def evaluate_rouge(prediction, reference):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=[prediction],
                         references=[reference],
                         use_aggregator=False,
                         )


def evaluate_bertscore(prediction, reference):
    bertscore = load('bertscore')
    return bertscore.compute(predictions=[prediction],
                             references=[reference],
                             lang='ru',
                             model_type='distilbert-base-uncased',
                             )


def evaluate_sacrebleu(prediction, reference):
    sacrebleu = evaluate.load('sacrebleu')
    return sacrebleu.compute(predictions=[prediction],
                             references=[reference],
                             tokenize='intl',
                             )


def evaluate_meteor(prediction, reference):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=[prediction],
                             references=[reference],
                             )
    return round(results['meteor'], 3)


def evaluate_bleu(prediction, reference):
    bleu = evaluate.load('bleu')
    return bleu.compute(predictions=[prediction],
                        references=[reference],
                        )


def calc_nlp_metrics(row):
    prediction = row['raw_text_t']
    reference = row['raw_text']

    result = evaluate_rouge(prediction, reference)
    row['rouge1'] = result['rouge1']
    row['rouge2'] = result['rouge2']
    row['rougeL'] = result['rougeL']
    row['rougeLsum'] = result['rougeLsum']

    result = evaluate_bertscore(prediction, reference)
    row['bertscore_pre'] = result['precision']
    row['bertscore_rec'] = result['recall']
    row['bertscore_f1'] = result['f1']

    row['sacrebleu'] = evaluate_sacrebleu(prediction, reference)['score']

    row['meteor'] = evaluate_meteor(prediction, reference)
    row['bleu'] = evaluate_bleu(prediction, reference)['bleu']


path_to_parsed_file = pathlib.Path('merged.xlsx')
df = pd.read_excel(path_to_parsed_file)

df = df.apply(calc_nlp_metrics, axis=1)

df.to_excel('RAG_metrics.xlsx')

