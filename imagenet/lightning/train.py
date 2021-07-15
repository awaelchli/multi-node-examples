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
r"""
This example is largely adapted from the PyTorch official examples repository
https://github.com/pytorch/examples/blob/master/imagenet/main.py and
and from the PyTorch Lightning Examples
https://github.com/PyTorchLightning/pytorch-lightning/blob/20f63377f81f4771d3f128f979b3a0f9b8d219a7/pl_examples/domain_templates/imagenet.py

Before you can run this example, you will need to download the ImageNet dataset manually from the
`official website <http://image-net.org/download>`_ and place it into a folder `path/to/imagenet`.

Train on ImageNet with default parameters:

.. code-block: bash

    python train.py --data.data-path /path/to/imagenet

or show all options you can change:

.. code-block: bash

    python train.py --help

Here is an example how to run on two nodes, 2 GPUs each:

.. code-block: bash

    # first node, Lightning launches two processes
    MASTER_ADDR=node01.cluster MASTER_PORT=1234 NODE_RANK=0 python train.py --trainer.gpus 2 --trainer.num_nodes 2 \
        --data.data-path ...

    # second node, Lightning launches two processes
    MASTER_ADDR=node02.cluster MASTER_PORT=1234 NODE_RANK=1 python train.py --trainer.gpus 2 --trainer.num_nodes 2 \
        --data.data-path ...
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
