# MindBeacon accuracy report

**Data:** 27,901 rows in the CSV; 27,837 usable student rows after cleaning (students only, valid sleep and diet entries, no missing financial stress). 58.5% are labelled depressed. No duplicate rows.

**Split:** 80% train / 20% test (5,568 unseen students), stratified, random_state=42. Cross-validation: 5-fold stratified on all rows.

## Model comparison
| Model | 5-fold CV accuracy | Test accuracy |
|---|---|---|
| Logistic regression (used) | 84.7% ± 0.6 | 85.0% |
| Gradient boosting | 84.5% ± 0.6 | 84.5% |
| Random forest | 84.4% ± 0.6 | 84.4% |

Logistic regression is used because it matches the more complex models, is explainable, and runs identically in the browser and the API.

## Final model on the unseen test set
| Metric | Value |
|---|---|
| Accuracy | 85.0% |
| Recall (depressed students found) | 88.7% |
| Precision | 86.1% |
| F1 | 87.4% |
| ROC-AUC | 0.924 |

## Confusion matrix (test set)
| | Predicted not depressed | Predicted depressed |
|---|---|---|
| **Actually not depressed** | 1842 | 468 (false alarms) |
| **Actually depressed** | 367 (missed) | 2891 |

## Decision threshold
| Threshold | Accuracy | Recall |
|---|---|---|
| 0.4 | 84.6% | 92.5% |
| 0.5 | 85.0% | 88.7% |
| 0.6 | 84.8% | 84.8% |

## Robustness
Without the "suicidal thoughts" question, cross-validated accuracy is **79.9%**. That one answer is the strongest single predictor in this dataset.

## Limits
Survey data shows patterns, not causes. About 85% is close to the ceiling for these features: tuned gradient boosting and random forests score about the same. The model is a screening aid, not a diagnosis.
