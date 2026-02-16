import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

filepath = 'logs/uq_eval/ambiginst.json'

with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)

print(len(content))

roc_list = []

refined_labels = []
ambig_uncertainty = []
unambig_uncertainty = []
uncertainty_list = []
for i in range(len(content)):
    ambig_flag = content[i]['isambig']
    if ambig_flag:
        refined_labels.append(True)
        ambig_uncertainty.append(content[i]['data_uncertainty'])
    else:
        refined_labels.append(False)
        unambig_uncertainty.append(content[i]['data_uncertainty'])
    uncertainty_list.append(content[i]['data_uncertainty'])

refined_labels = np.array(refined_labels)
ys_array = refined_labels
xs_array = np.array(uncertainty_list)
print(xs_array.shape)
auroc = roc_auc_score(refined_labels, xs_array)
print("auroc:", auroc)
print()
roc_list.append(auroc)

print(roc_list)
print(np.mean(roc_list))

print("ambig: ", np.mean(ambig_uncertainty))
print("unambig: ", np.mean(unambig_uncertainty))

all_f1s = []
all_precisions = []
all_recalls = []
all_accuracies = []  # NEW: track accuracy per threshold

thres_cdts = np.arange(1,100) / 100
for thres in thres_cdts:
    pred_correctness_labels = np.array([x > thres for x in xs_array])
    tgt_correctness_labels = ys_array

    corr_f1 = f1_score(tgt_correctness_labels, pred_correctness_labels)
    precision = precision_score(tgt_correctness_labels, pred_correctness_labels)
    recall = recall_score(tgt_correctness_labels, pred_correctness_labels)
    acc = accuracy_score(tgt_correctness_labels, pred_correctness_labels)  # NEW

    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(corr_f1)
    all_accuracies.append(acc)  # NEW

best_f1_idx = np.argmax(all_f1s)
print("best f1: ", all_f1s[best_f1_idx])
print('best precision: ', all_precisions[best_f1_idx])
print("best recall: ", all_recalls[best_f1_idx])
print("best thres (by f1): ", thres_cdts[best_f1_idx])

# Accuracy at the best-F1 threshold (overall)
acc_at_best_f1 = all_accuracies[best_f1_idx]
print("accuracy at best-f1 thres: ", acc_at_best_f1)

# Also report the best accuracy and its threshold (optional but handy)
best_acc_idx = np.argmax(all_accuracies)
print("best accuracy: ", all_accuracies[best_acc_idx])
print("best thres (by accuracy): ", thres_cdts[best_acc_idx])

best_thres = thres_cdts[best_f1_idx]

ambig_preds = np.array([x > best_thres for x in ambig_uncertainty])
unambig_preds = np.array([x <= best_thres for x in unambig_uncertainty])

# Per-group accuracies at the chosen threshold
ambig_pred_acc = np.sum(ambig_preds) / len(ambig_uncertainty) if len(ambig_uncertainty) > 0 else float('nan')
print("ambig acc: ", ambig_pred_acc)

unambig_pred_acc = np.sum(unambig_preds) / len(unambig_uncertainty) if len(unambig_uncertainty) > 0 else float('nan')
print("unambig acc: ", unambig_pred_acc)

# Overall accuracy at the chosen threshold (same as acc_at_best_f1, printed again for clarity)
overall_preds = np.array([x > best_thres for x in xs_array])
overall_acc = accuracy_score(ys_array, overall_preds)
print("overall acc (best-f1 thres): ", overall_acc)
