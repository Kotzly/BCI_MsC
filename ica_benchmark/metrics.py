from sklearn.metrics import recall_score, f1_score, precision_score


def micro_precision(*args):
    return precision_score(*args, average="micro")


def macro_precision(*args):
    return precision_score(*args, average="macro")


def micro_recall(*args):
    return recall_score(*args, average="micro")


def macro_recall(*args):
    return recall_score(*args, average="macro")


def micro_f1(*args):
    return f1_score(*args, average="micro")


def macro_f1(*args):
    return f1_score(*args, average="macro")
