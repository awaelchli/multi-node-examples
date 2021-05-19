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

    python train.py --data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python train.py --help

"""

from pytorch_lightning.utilities.cli import LightningCLI

from data import ImageNetDataModule
from model import ImageNetLightningModel


def main():
    cli = LightningCLI(
        description="PyTorch Lightning ImageNet Training",
        model_class=ImageNetLightningModel,
        datamodule_class=ImageNetDataModule,
        seed_everything_default=123,
        trainer_defaults=dict(
            accelerator="ddp",
            max_epochs=90,
        ),
    )
    # TODO: determine per-process batch size given total batch size
    # TODO: enable evaluate
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
