import torch
from pytorch_lightning import LightningModule
from torch import optim as optim
from torch.nn import functional as F
from torch.optim import lr_scheduler as lr_scheduler
from torchvision import models as models


class ImageNetLightningModel(LightningModule):
    def __init__(
        self,
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.model = models.resnet18(pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, logger=True)
        self.log(
            "train_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True
        )
        self.log("train_acc5", acc5, on_step=True, on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log("val_loss", loss_val, on_step=True, on_epoch=True)
        self.log("val_acc1", acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("val_acc5", acc5, on_step=True, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        outputs = self.validation_epoch_end(*args, **kwargs)

        def substitute_val_keys(out):
            return {k.replace("val", "test"): v for k, v in out.items()}

        outputs = {
            "test_loss": outputs["val_loss"],
            "progress_bar": substitute_val_keys(outputs["progress_bar"]),
            "log": substitute_val_keys(outputs["log"]),
        }
        return outputs

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("ImageNetLightningModel")
        parser.add_argument(
            "-j",
            "--workers",
            default=4,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 4)",
        )
        parser.add_argument(
            "-b",
            "--batch-size",
            default=256,
            type=int,
            metavar="N",
            help="mini-batch size (default: 256), this is the total batch size of all GPUs on the current node"
            " when using Data Parallel or Distributed Data Parallel",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.1,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            action="store_true",
            help="use pre-trained model",
        )
        parser.add_argument(
            "--fake-data",
            default=False,
            action="store_true",
            help="simulate fake data instead of using ImageNet",
        )
        return parent_parser
