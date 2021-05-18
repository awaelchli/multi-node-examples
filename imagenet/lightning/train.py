# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This example is largely adapted from the PyTorch official examples repository
https://github.com/pytorch/examples/blob/master/imagenet/main.py and
and from the PyTorch Lightning Examples
https://github.com/PyTorchLightning/pytorch-lightning/blob/20f63377f81f4771d3f128f979b3a0f9b8d219a7/pl_examples/domain_templates/imagenet.py

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python imagenet.py --data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python imagenet.py --help

"""

from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl

from .model import ImageNetLightningModel
from .data import ImageNetDataModule


def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.accelerator == "ddp":
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = ImageNetLightningModel(**vars(args))
    datamodule = ImageNetDataModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.test(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule)


def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        "--data-path", metavar="DIR", type=str, help="path to dataset"
    )
    parent_parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parent_parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    parser = ImageNetLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        accelerator="ddp",
        profiler="simple",
        max_epochs=90,
    )
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_cli()
