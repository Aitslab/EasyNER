import pandas as pd
import warnings
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


def read_infile(infile):
    with open(infile, encoding="utf8") as f:
        return f.readlines()
    
def generate_classification_report(pred_file, true_file, outfile, pred_sep=" ", true_sep=" "):
    y_true = []
    y_pred = []
    pred_file_lines = read_infile(pred_file)
    true_file_lines = read_infile(true_file)

    if len(pred_file_lines)!= len(true_file_lines):
        raise Exception("Err! Prediction file and annotated file mismatch!")
    

    for line_p, line_t in zip(pred_file_lines, true_file_lines):
        line_p = line_p.strip().split(pred_sep)
        line_t = line_t.strip().split(true_sep)

        if line_p[0]!=line_t[0]:
            warnings.warn("Possible file mismath detected. Check input files!")
        else:
            try:
                word, yp = line_p
                yt = line_t[1]
                y_true.append(yt)
                y_pred.append(yp)
            except:
                pass
    with open(outfile, "w", encoding="utf8") as f:
        f.write(classification_report([y_true],[y_pred], digits=5))

def get_metrics(metrics_config):
    pred_sep = metrics_config["pred_sep"] if "pred_sep" in metrics_config else " "
    true_sep = metrics_config["true_sep"] if "true_sep" in metrics_config else " "
    
    generate_classification_report(metrics_config["predictions_file"],
                                    metrics_config["true_file"],
                                    metrics_config["output_file"],
                                    pred_sep=pred_sep, true_sep=true_sep)

if __name__=="__main__":

    pred_file = "../../NER_pipeline/results_testeval_p50/test_results/huner/huner_cell/test.txt"
    true_file = "../../NER_pipeline/results_testeval_p50/test_results/huner/huner_cell/test_predictions.txt"
    outfile = "../temp/huner_cell_cr.txt"

    generate_classification_report(pred_file, true_file, outfile)
