import sklearn.metrics as perf
import torch
import sys
sys.path.insert(0,'..')
from utils.preprocessing import get_buckets, assign_bucket


def get_predictions(model=None, loader=None, model_type=None, task=None,
                    curr_buckets=None, max_usefulCount=None):
    predictions = []
    gt = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            if task == 'CLASSIFICATION':
                target = target.long()
            if model_type.upper() == 'META':
                output = model(nonText).squeeze(1)
            elif model_type.upper() == 'TEXT':
                output = model(tokens, attention_mask).squeeze(1)
            elif model_type.upper() == 'TEXT-META':
                output = model(tokens, attention_mask, nonText)
            else:
                print('Invalid model_type: should be meta, text, or text-meta')

            if task.upper() == 'REGRESSION':
                predictions.extend(list(output.cpu()))
            elif task.upper() == 'CLASSIFICATION':
                predictions.extend(list(torch.argmax(output, axis=1).cpu()))
            elif task.upper() == 'ORDINAL':
                predictions.extend(get_ordinal_prediction(output, curr_buckets, max_usefulCount))
            else:
                print('Invalid task: should be regression or classification')
            gt.extend(list(target.cpu()))

    return predictions, gt


def get_ordinal_prediction(reg_pred=None, curr_buckets=None, max_usefulCount=None):
    pred_usefulCount = reg_pred*max_usefulCount  # model predicts 0 to 1, so we need to get prediction in terms of counts for assign_bucket
    class_pred = list(pred_usefulCount.cpu().apply_(lambda x: assign_bucket(x, curr_buckets)))
    return class_pred


def get_ordinal_cls_perf(model=None, loader=None, model_type=None, curr_buckets=None,
                         max_usefulCount=None):
    pred, gt = get_predictions(model, loader, model_type, 'ORDINAL', curr_buckets, max_usefulCount)
    f1 = perf.f1_score(gt, pred, average='macro')
    acc = perf.accuracy_score(gt, pred)

    return f1, acc


def get_cls_perf(model=None, loader=None, model_type=None):
    pred, gt = get_predictions(model, loader, model_type, 'CLASSIFICATION')
    f1 = perf.f1_score(gt, pred, average='macro')
    acc = perf.accuracy_score(gt, pred)

    return f1, acc


def get_reg_perf(model=None, loader=None, model_type=None):
    pred, gt = get_predictions(model, loader, model_type, 'REGRESSION')
    r2 = perf.r2_score(gt, pred)
    rmse = perf.mean_squared_error(gt, pred, squared=False)
    mae = perf.mean_absolute_error(gt, pred)

    return mae, rmse, r2
