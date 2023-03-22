from collections import defaultdict, Counter
from typing import DefaultDict, List, Counter as CounterT
from scripts.ner_inference import NERInferenceSession_biobert_onnx
import conlleval
import sys


def gs_metrics(input_path: str):
    with open(input_path, "r") as f:
        data = f.readlines()

    label_count = defaultdict(int)
    occurrence_count = defaultdict(int)
    occurrences = 0

    for line in data:
        line = line.strip()

        if line:
            line = line.split()[1]
            label_count[line] += 1

            if line == "B":
                occurrences += 1

        else:
            occurrence_count[occurrences] += 1
            occurrences = 0

    print(" - - - Gold standard metrics - - -")

    print("Label count:")
    for key in label_count:
        print("\t" + key + " label count: " + str(label_count[key]))

    print("\nOccurrence count:")
    for key in sorted(occurrence_count):
        print("\t" + str(key) + "_occurrence count: " + str(occurrence_count[key]))

    print(" - - - - - - - - - - - - - - - - - \n")

def sentence_metrics(pred_labels: List[str], gs_labels: List[str]):

    # Treating B = I
    confusion_matrix = defaultdict(int)
    for pred, gs in zip(pred_labels, gs_labels):

        if pred == "B" or pred == "I":
            if gs == "B" or gs == "I":
                confusion_matrix["true_positive"] += 1
            elif gs == "O":
                confusion_matrix["false_positive"] += 1
        elif pred == "O":
            if gs == "O":
                confusion_matrix["true_negative"] += 1
            elif gs == "B" or gs == "I":
                confusion_matrix["false_negative"] += 1

    # Treating B=/=I
    token_matrix = defaultdict(lambda: defaultdict(int))

    for pred, gs in zip(pred_labels, gs_labels):
        token_matrix[gs][pred] += 1

    return confusion_matrix, token_matrix


def biobert_metrics(model: NERInferenceSession_biobert_onnx, input_path: str, output_path: str):
    with open(input_path, "r") as f:
        data = f.readlines()

    total = 0
    for i in data:
        if i == "\n":
            total += 1

    print("Running over " + str(total) + " sentences")

    confusion_matrix: CounterT[str] = Counter()
    token_matrix: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    gs_labels: List[str] = []
    sequence = ""
    line_list = list()

    counter_2 = 0

    for line in data:

        if line == "\n":
            counter_2 += 1

            sys.stdout.write("Predicted {}/{} sentences so far.\r".format(counter_2, total))
            sys.stdout.flush()

            pred_pairs = model.predict(sequence.strip())

            tokens = sequence.strip().split()

            # The tokenization label X and special labels hold no more value
            pred_labels = [label[1] for label in pred_pairs if label[1]
                           != 'X' and label[0] != '[CLS]' and label[0] != '[SEP]']

            cm, tm = sentence_metrics(pred_labels, gs_labels)

            confusion_matrix.update(cm)

            for gs_label in tm:
                for pred_label in tm[gs_label]:
                    token_matrix[gs_label][pred_label] += tm[gs_label][pred_label]


            line_list = line_list + list(map(lambda token, gs, pred: token + " TK " + gs + " " + pred, tokens, gs_labels, pred_labels))

            gs_labels = []
            sequence = ""
            continue

        columns = line.split("\t")
        sequence += columns[0] + " "
        gs_labels.append(columns[1].strip())

        #if counter_2 == 1000:
            #break

    conlleval_res = conlleval.report(conlleval.evaluate(line_list))
    print(conlleval_res)

    # CM
    cm_r = confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_negative"])
    cm_p = confusion_matrix["true_positive"]/(confusion_matrix["true_positive"] + confusion_matrix["false_positive"])
    cm_f1 = 2*cm_r*cm_p / (cm_r + cm_p)

    # TM
    b_r = token_matrix["B"]["B"] / (token_matrix["B"]["B"] + token_matrix["B"]["I"] + token_matrix["B"]["O"])
    b_p = token_matrix["B"]["B"] / (token_matrix["B"]["B"] + token_matrix["I"]["B"] + token_matrix["O"]["B"])
    b_f1 = 2*b_r*b_p / (b_r + b_p)

    i_r = token_matrix["I"]["I"] / (token_matrix["I"]["B"] + token_matrix["I"]["I"] + token_matrix["I"]["O"])
    i_p = token_matrix["I"]["I"] / (token_matrix["B"]["I"] + token_matrix["I"]["I"] + token_matrix["O"]["I"])
    i_f1 = 2*i_r*i_p / (i_r + i_p)

    o_r = token_matrix["O"]["O"] / (token_matrix["O"]["B"] + token_matrix["O"]["I"] + token_matrix["O"]["O"])
    o_p = token_matrix["O"]["O"] / (token_matrix["B"]["O"] + token_matrix["I"]["O"] + token_matrix["O"]["O"])
    o_f1 = 2*o_r*o_p / (o_r + o_p)

    with open(output_path, "a+") as out_f:
        out_f.write("\nConlleval results:\n" + conlleval_res)

        out_f.write("\nToken-Level Confusion Matrix:\n"
                    + "True Positive:\t" + str(confusion_matrix["true_positive"])
                    + "\nTrue Negative:\t" + str(confusion_matrix["true_negative"])
                    + "\nFalse Positive:\t" + str(confusion_matrix["false_positive"])
                    + "\nFalse Negative:\t" + str(confusion_matrix["false_negative"])
                    + "\nRecall:\t\t" + str(cm_r)
                    + "\nPrecision:\t" + str(cm_p)
                    + "\nF1-score:\t" + str(cm_f1))

        out_f.write("\n\nToken Matrix (true\predicted):\n\tB\tI\tO\n"
                    + "B\t" + str(token_matrix["B"]["B"]) + "\t" + str(token_matrix["B"]["I"]) + "\t" + str(token_matrix["B"]["O"])
                    + "\nI\t" + str(token_matrix["I"]["B"]) + "\t" + str(token_matrix["I"]["I"]) + "\t" + str(token_matrix["I"]["O"])
                    + "\nO\t" + str(token_matrix["O"]["B"]) + "\t" + str(token_matrix["O"]["I"]) + "\t" + str(token_matrix["O"]["O"])
                    + "\nB_Recall:\t" + str(b_r)
                    + "\nB_Precision:\t" + str(b_p)
                    + "\nB_F1:\t\t" + str(b_f1)
                    + "\nI_Recall:\t" + str(i_r)
                    + "\nI_Precision:\t" + str(i_p)
                    + "\nI_F1:\t\t" + str(i_f1)
                    + "\nO_Recall:\t" + str(o_r)
                    + "\nO_Precision:\t" + str(o_p)
                    + "\nO_F1:\t\t" + str(o_f1) + "\n")



    print("Confusion matrix:")
    print({**confusion_matrix})
    print("Recall: " + str(cm_r))
    print("Precision: " + str(cm_p))
    print()

    print("Token matrix:")
    print({**token_matrix})
    print()