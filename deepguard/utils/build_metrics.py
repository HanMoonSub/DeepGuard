from torchmetrics import (
                            MetricCollection, 
                            Accuracy, 
                            AUROC, 
                            Precision, 
                            Recall, 
                            F1Score
                        )

def build_metrics(
    device: str = "cpu",
    task: str = "binary"
):
    metrics = MetricCollection({
        'acc': Accuracy(task=task),
        'auc': AUROC(task=task),
        'f1': F1Score(task=task),
        'precision': Precision(task=task),
        'recall': Recall(task=task)
    })
        
    return metrics.to(device)
    