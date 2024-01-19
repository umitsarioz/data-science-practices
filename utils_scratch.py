def calculate_confusion_matrix(y_true, y_pred) -> tuple:
    """
    Confusion Matrix  = [[TP,   FP]

                         [FN,   FP]]
    """
    from sklearn.metrics import confusion_matrix
    cfsm = confusion_matrix(y_true, y_pred)
    TP, TN, FP, FN = cfsm[0][0], cfsm[1][1], cfsm[0][1], cfsm[1][0]

    return TP, TN, FP, FN


def calculate_error(TP, TN, FP, FN) -> float:
    """
    Error : (FP + FN) / (TP+TN+FP+FN)
    """
    error = (FP + FN) / (TP + TN + FP + FN)
    return error


def calculate_accuracy(TP, TN, FP, FN) -> float:
    """
    Accuracy : (TP + TN) / (TP+TN+FP+FN)
    """
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


def calculate_precision(TP, FP) -> float:
    """
    Precision : TP / (FP + TP)
    """
    precision = TP / (FP + TP)
    return precision


def calculate_recall_sensitivity(TP, FN) -> float:
    """
    Recall : TP / (FN + TP)
    """
    recall = TP / (FN + TP)
    return recall


def calculate_specificity(TN, FP) -> float:
    """
    Specificity : TN / (TN + FP)
    """
    specificity = TN / (TN + FP)
    return specificity


def calculate_f1(precision, recall) -> float:
    """
    f1 = (2*precision*recall) / (precision+recall)
    """
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_and_show_rates(TP, TN, FP, FN) -> tuple:
    Accuracy = calculate_accuracy(TP, TN, FP, FN)
    Error = calculate_error(TP, TN, FP, FN)
    Precision = calculate_precision(TP, FP)
    Recall = calculate_recall_sensitivity(TP, FN)
    Sensitivity = calculate_recall_sensitivity(TP, FN)
    Specificity = calculate_specificity(TN, FP)
    F1_score = calculate_f1(Precision, Recall)
    print(f"Accuracy : {Accuracy:.3f}")
    print(f"Error Rate: {Error:.3f}")
    print(f"Precision: {Precision:.3f}")
    print(f"Recall: {Recall:.3f}")
    print(f"Sensitivity: {Sensitivity:.3f}")
    print(f"Specificity: {Specificity:.3f}")
    print(f"F1 Score: {F1_score:.3f}")
    return Accuracy, Error, Precision, Recall, Sensitivity, Specificity, F1_score
