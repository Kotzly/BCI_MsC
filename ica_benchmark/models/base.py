import torch
from torch import nn
from ica_benchmark.metrics import micro_f1, micro_precision, micro_recall
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
from lightning.pytorch import LightningModule


class LightningEEGModule(LightningModule):

    def set_trainer(self, trainer):
        self.trainer = trainer

    def __init__(self, *args, **kwargs):
        super(LightningEEGModule, self).__init__(*args, **kwargs)
        self.test_step_outputs = list()
        self.test_step_labels = list()
        self.val_step_outputs = list()
        self.val_step_labels = list()
        self.train_step_outputs = list()
        self.train_step_labels = list()

    def loss(self, y_pred, y_true):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_true)
        return loss

    def calculate_metrics(self, y_pred, y_true, name=None):

        if (y_pred.ndim > 2) or ((y_pred.ndim == 2) and (y_pred.size(1) > 1)):
            y_pred = y_pred.argmax(axis=1)

        if y_pred.is_cuda:
            y_pred = y_pred.cpu()
        if y_true.is_cuda:
            y_true = y_true.cpu()

        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        metric_fns = (
            cohen_kappa_score,
            balanced_accuracy_score,
            micro_f1,
            micro_precision,
            micro_recall
        )
        metrics = {
            metric_fn.__name__: metric_fn(y_true, y_pred)
            for metric_fn
            in metric_fns
        }
        if name:
            metrics = {
                f"{name}_{k}": v
                for k, v
                in metrics.items()
            }
        return metrics

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        output = self(batch_x)
        metric_dict = {
            "loss": self.loss(output, batch_y)
        }
        self.train_step_outputs.append(output)
        self.train_step_labels.append(batch_y)
        return metric_dict
#     def on_train_epoch_end(self)

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        output = self(batch_x)
        self.val_step_outputs.append(output)
        self.val_step_labels.append(batch_y)

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        output = self(batch_x)
        self.test_step_outputs.append(output)
        self.test_step_labels.append(batch_y)

    def on_train_epoch_end(self, *args):
        metric_dict = self.calculate_metrics(
            torch.cat(self.train_step_outputs),
            torch.cat(self.train_step_labels),
            name="train"
        )

        metric_dict["train_loss"] = self.loss(
            torch.cat(self.train_step_outputs),
            torch.cat(self.train_step_labels)
        )
        self.log_dict(metric_dict)

        self.train_step_outputs.clear()
        self.train_step_labels.clear()

    def on_validation_epoch_end(self, *args):
        metric_dict = self.calculate_metrics(
            torch.cat(self.val_step_outputs),
            torch.cat(self.val_step_labels),
            name="val"
        )

        metric_dict["val_loss"] = self.loss(
            torch.cat(self.val_step_outputs),
            torch.cat(self.val_step_labels)
        )
        self.log_dict(metric_dict)

        self.val_step_outputs.clear()
        self.val_step_labels.clear()

    def on_test_epoch_end(self, *args):
        metric_dict = self.calculate_metrics(
            torch.cat(self.test_step_outputs),
            torch.cat(self.test_step_labels),
            name="test"
        )
        self.log_dict(metric_dict)

        self.test_step_outputs.clear()
        self.test_step_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def fit(self, train_dataloaders, val_dataloaders=None):
        self.trainer.fit(
            self,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
        )

    def predict(self, *args, **kwargs):
        pred = self(*args, **kwargs)
        return pred.argmax(axis=1)

    def test(self, test_dataloaders):
        return self.trainer.test(self, test_dataloaders)
