from typing import Dict
import torch
from mmm.data_loading.ClassificationDataset import ClassificationDataset
from mmm.data_loading.TrainValCohort import TrainValCohort
from mmm.data_loading.MTLDataset import mtl_batch_collate
from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset
from mmm.mtl_modules.shared_blocks.SharedBlock import SharedBlock
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.tasks.TaskModule import TaskModule
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.Squeezer import Squeezer
from mmm.mtl_modules.shared_blocks.Grouper import Grouper
from mmm.neural.modules.TorchVisionCNN import TorchVisionCNN


def test_task_module():
    mockup_dataset = ClassificationDataset(
        ClassificationMockupDataset(N=10, image_shape=(3, 64, 64)),
        class_names=["a", "b"],
    )
    encoder = PyramidEncoder(PyramidEncoder.Config(model=TorchVisionCNN.Config()))
    squeezer = Squeezer(
        args=Squeezer.Config(out_channels=-1),
        enc_out_channels=encoder.get_feature_pyramid_channels(),
        enc_strides=encoder.get_strides(),
    )
    encoder.set_active_task("original")
    encoder.eval()
    shared_modules: Dict[str, SharedBlock] = {"encoder": encoder, "squeezer": squeezer}
    task: MTLTask = ClassificationTask(
        squeezer.get_hidden_dim(),
        ClassificationTask.Config(module_name="test_task"),
        TrainValCohort(TrainValCohort.Config(batch_size=(2, 2)), mockup_dataset, None),
    )
    task.eval()
    task_module = TaskModule(task, shared_modules)

    with torch.no_grad():
        single_item = mockup_dataset[0]["image"]
        # Second element are the supercase indices
        network_input = (torch.stack([single_item, single_item]), None)

        y_shouldbe = task.forward(network_input, shared_modules)
        y_taskmodule = task_module(network_input)

        assert torch.equal(y_shouldbe, y_taskmodule)


def test_mil_clf_task():
    mockup_dataset = ClassificationDataset(
        ClassificationMockupDataset(N=2, image_shape=(12, 3, 64, 64)),
        class_names=["a", "b"],
        batch_transform=lambda x: [{"image": c, "class": 1, "meta": {"group_id": "0"}} for c in x["image"]],
        collate_fn=mtl_batch_collate,
    )
    encoder = PyramidEncoder(PyramidEncoder.Config(model=TorchVisionCNN.Config()))
    squeezer = Squeezer(
        args=Squeezer.Config(out_channels=-1),
        enc_out_channels=encoder.get_feature_pyramid_channels(),
        enc_strides=encoder.get_strides(),
    )
    grouper = Grouper(
        Grouper.Config(module_name="grouper", version="weighted"),
        embedding_dim=encoder.get_feature_pyramid_channels()[-1],
    )

    encoder.set_active_task("original")
    encoder.eval()
    shared_modules: Dict[str, SharedBlock] = {
        "encoder": encoder,
        "squeezer": squeezer,
        "grouper": grouper,
    }
    task: MTLTask = ClassificationTask(
        squeezer.get_hidden_dim(),
        ClassificationTask.Config(module_name="test_task", grouper_key="grouper"),
        TrainValCohort(TrainValCohort.Config(batch_size=(1, 1)), mockup_dataset, None),
    )
    task.eval()
    blocks = SharedModules(shared_modules)

    with torch.no_grad():
        dl = mockup_dataset.get_dataloader()
        single_item = next(iter(dl))

        # Can item pass through the whole training process without causing an error
        _ = task.training_step(batch=single_item, shared_blocks=blocks)
