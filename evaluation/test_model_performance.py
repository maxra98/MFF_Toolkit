import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier, Pool
from scipy import stats

# Function to compute confidence intervals using normal distribution
def compute_confidence_interval(metric_values):
    mean = np.mean(metric_values)
    std_dev = np.std(metric_values, ddof=1)
    confidence_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev / np.sqrt(len(metric_values)))
    return mean, confidence_interval

def cross_validate_model(X, y, n_splits=10):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_preds = []
    all_labels = []
    all_probs = []
    cm_scores = []
    roc_aucs = []
    precision_scores = []
    specificity_scores = []
    recall_scores = []
    f1_scores = []
    confusion_matrices = np.zeros((2, 2))  # For 3-class problem

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'Fold {fold + 1}/{n_splits}')
        
        # Prepare train and validation data
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Check dimensions
        assert X_train.shape[0] == y_train.shape[0], f"X_train and y_train have different number of samples for fold {i}"
        assert X_val.shape[0] == y_val.shape[0], f"X_test and y_test have different number of samples for fold {i}"

        train_pool = Pool(data=X_train, label=y_train)
        test_pool = Pool(data=X_val, label=y_val)

        catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=2, verbose=10)
        catboost_model.fit(train_pool)

        y_pred = catboost_model.predict(test_pool)
        y_pred_prob = catboost_model.predict_proba(test_pool)

        preds = y_pred
        labels = y_val

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(y_pred_prob)

        # Update confusion matrix
        cm = confusion_matrix(labels, preds)
        confusion_matrices += cm
        cm_scores.append(cm)

        tn, fp, fn, tp = cm.ravel()
        
        # Compute ROC AUC for this fold
        fpr, tpr, _ = roc_curve(labels, y_pred_prob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        # Compute precision, recall, specificity and f1-score
        precision = precision_score(labels, preds, average='macro')
        specificity = tn / (tn + fp)
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        precision_scores.append(precision)
        specificity_scores.append(specificity)
        recall_scores.append(recall)
        f1_scores.append(f1)

    # Aggregate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Compute classification report with confidence intervals
    precision_mean, precision_ci = compute_confidence_interval(precision_scores)
    specificity_mean, specificity_ci = compute_confidence_interval(specificity_scores)
    recall_mean, recall_ci = compute_confidence_interval(recall_scores)
    f1_mean, f1_ci = compute_confidence_interval(f1_scores)
    roc_auc_mean, roc_auc_ci = compute_confidence_interval(roc_aucs)

    print(f'Precision: {precision_mean:.4f} ± {precision_ci[1]-precision_mean:.4f}')
    print(f'Specificity: {specificity_mean:.4f} ± {specificity_ci[1]-specificity_mean:.4f}')
    print(f'Recall: {recall_mean:.4f} ± {recall_ci[1]-recall_mean:.4f}')
    print(f'F1-Score: {f1_mean:.4f} ± {f1_ci[1]-f1_mean:.4f}')
    print(f'ROC AUC: {roc_auc_mean:.4f} ± {roc_auc_ci[1]-roc_auc_mean:.4f}')

    # Print final confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices.astype(int))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return all_labels, all_preds, all_probs, roc_aucs, confusion_matrices, cm_scores

def plot_roc_and_pr_curves(labels, preds):
    # ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pr_auc = dict()
    n_classes = 2
    class_names = ["NORM", "AF"]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((labels == i).astype(int), preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve((labels == i).astype(int), preds[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f'PR curve for class {class_names[i]} (area = {pr_auc[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
