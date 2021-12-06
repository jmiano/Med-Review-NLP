import sklearn.metrics as perf
import torch


def get_predictions(model=None, loader=None, model_type=None, task=None):
    predictions = []
    gt = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
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
            else:
                print('Invalid task: should be regression or classification')
            gt.extend(list(target.cpu()))

    return predictions, gt


def get_cls_perf(model=None, loader=None, model_type=None):
    pred, gt = get_predictions(model, loader, model_type, 'CLASSIFICATION')
    f1 = perf.f1_score(gt, pred, average='macro')
    acc = perf.accuracy_score(gt, pred)

    return f1, acc


def get_reg_perf(model=None, loader=None, model_type=None):
    pred, gt = get_predictions(model, loader, model_type, 'REGRESSION')
    r2 = perf.r2_score(gt, pred)
    rmse = perf.mean_squared_error(gt, pred, squared=False)

    return rmse, r2
