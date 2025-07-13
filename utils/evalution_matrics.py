import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import jiwer

               ##Voice Activity Detection (VAD) Evaluation Metrics##


def evaluate_vad(y_true, y_pred):
    """
    Evaluate classification performance for VAD .
    Args:
        y_true (list or np.array): Ground truth binary labels (0: silence, 1: voice)
        y_pred (list or np.array): Predicted binary labels

    Returns:
        dict: accuracy, precision, recall, f1_score
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

                              ## TRIGGER WORD DETECTION EVALUTION

def evalute_trigger(y_true, y_pred):
    """
    Evaluate classification performance for trigger word detection.
    Args:
        y_true (list or np.array): Ground truth binary labels (0 or 1)
        y_pred (list or np.array): Predicted binary labels (0 or 1)

    Returns:
        dict: precision, recall, f1_score
    """
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }

                                ##  AUTOMATED SPEECH RECOGNITION ASR EVALUATION METRICS ##
def evaluate_asr(references, hypotheses):
    """
    Compute Word Error Rate (WER) for ASR.

    Args:
        references (list of str): Ground truth transcripts
        hypotheses (list of str): Predicted transcripts

    Returns:
        dict: Word Error Rate (WER) in %
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    wer = jiwer.wer(references, hypotheses,
                    truth_transform=transformation,
                    hypothesis_transform=transformation)

    return {
        "WER (%)": round(wer * 100, 2)
    }
