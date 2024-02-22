from torch.utils.data import DataLoader

from mmm.data_loading.synthetic.mockup import ClassificationMockupDataset


def test_classification_mockup_ds():
    ds = ClassificationMockupDataset(5)
    element = ds.__getitem__(1)
    assert element["image"].max() == element["image"].min() == 1.0
    dl = DataLoader(ds, batch_size=8)
    d = next(iter(dl))
    assert d["image"].size() == (5, 1, 64, 64) and d["class"].size() == (5, 1)
