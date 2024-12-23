import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score

# Path to CSV with predictions and references
PREDICTIONS_FILE = "data/output_datasets/predictions.csv"  

# define performance metrics 
def compute_bleu(predictions, references):
    bleu_scores = [sentence_bleu([ref.split()], pred.split()) for pred, ref in zip(predictions, references)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return bleu_scores, avg_bleu

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    avg_rouge = {
        key: sum(score[key].fmeasure for score in rouge_scores) / len(rouge_scores)
        for key in ['rouge1', 'rouge2', 'rougeL']
    }
    return avg_rouge

def compute_exact_match(predictions, references):
    return accuracy_score(references, predictions)

def compute_bert_score(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    avg_bert_score = F1.mean().item()
    return F1.tolist(), avg_bert_score

def evaluate(predictions_file):
    df = pd.read_csv(predictions_file)
    if 'predicted' not in df.columns or 'reference' not in df.columns:
        raise ValueError("The CSV file must contain 'predicted' and 'reference' columns.")

    predictions = df['predicted'].astype(str).tolist()
    references = df['reference'].astype(str).tolist()

    print(f"Loaded {len(predictions)} predictions and references.")

    # compute metrics
    bleu_scores, avg_bleu = compute_bleu(predictions, references)
    avg_rouge = compute_rouge(predictions, references)
    em_score = compute_exact_match(predictions, references)
    bert_scores, avg_bert_score = compute_bert_score(predictions, references)

    # show results 
    print("\nEvaluation Results:")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average ROUGE Scores: {avg_rouge}")
    print(f"Exact Match (EM): {em_score:.4f}")
    print(f"Average BERTScore (F1): {avg_bert_score:.4f}")

    # save results 
    df['bleu_score'] = bleu_scores
    df['bert_score'] = bert_scores
    df.to_csv("data/output_datasets/evaluation_results.csv", index=False)
    print("\nDetailed results saved to 'data/output_datasets/evaluation_results.csv'.")

if __name__ == "__main__":
    print("Starting evaluation...")
    evaluate(PREDICTIONS_FILE)
