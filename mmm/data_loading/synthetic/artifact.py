from typing import Dict, Any
import torch
import torchvision
import cv2 as cv
from monai.transforms import Transform


class RandomLineAugmentation(Transform):
    def __call__(self, data: torch.Tensor):
        """
        Expect a normlized RGB Torch Tensor
        """

        _, data_x, data_y = data.squeeze().shape
        # generate random points (without replacement)
        rnd_x = torch.randperm(data_x)[:2]
        rnd_y = torch.randperm(data_y)[:2]

        # transfrom into cv2 readable landmarks
        pt1 = (rnd_x[0].item(), rnd_y[0].item())
        pt2 = (rnd_x[1].item(), rnd_y[1].item())

        # create artifact mask
        artifacts = torch.ones((data_x, data_y, 1), dtype=torch.uint8).numpy() * 255.0
        line_mask = cv.line(artifacts, pt1, pt2, color=(0, 0, 0), thickness=data_x // 32) / 255.0

        rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
        # (data_x//rnd_blurr_divider, data_y//rnd_blurr_divider))
        blurred_mask = cv.blur(line_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))

        # multiply input img with mask to create artifact
        return data * blurred_mask


class RandomBlobAugmentation(Transform):
    def __call__(self, data: torch.Tensor):
        _, data_x, data_y = data.squeeze().shape
        # generate random points
        min_radius = data_x // 30
        rnd_x = torch.randint(min_radius, data_x - min_radius, size=(1,))
        rnd_y = torch.randint(min_radius, data_y - min_radius, size=(1,))
        center = (rnd_x.item(), rnd_y.item())
        rnd_size_divider = torch.randint(7, 20, size=(1,)).item()

        artifacts = torch.ones((data_x, data_y, 1), dtype=torch.uint8).numpy() * 255.0
        circle_mask = (
            cv.circle(
                artifacts,
                center,
                radius=data_x // rnd_size_divider,
                color=(0, 0, 0),
                thickness=-1,
            )
            / 255.0
        )

        rnd_blurr_divider = torch.randint(15, 20, size=(1,)).item()
        blurred_mask = cv.blur(circle_mask, (data_x // rnd_blurr_divider, data_y // rnd_blurr_divider))
        return data * blurred_mask


class RandomTearAugmentation(Transform):
    def __call__(self, data: torch.Tensor):
        _, data_x, data_y = data.squeeze().shape
        data = data.squeeze()
        # generate random points
        x_shift = data_x // 30
        rnd_x, _ = torch.sort(torch.randperm(data_x - x_shift)[:2])
        rnd_y, _ = torch.sort(torch.randperm(data_y)[:2])
        while rnd_x[1] - rnd_x[0] < 10:
            rnd_x, _ = torch.sort(torch.randperm(data_x - x_shift)[:2])

        crop = data[:, rnd_x[0] : rnd_x[1], rnd_y[0] : rnd_y[1]]
        mask = torch.zeros((1, data_x, data_y))
        mask[:, rnd_x[0] : rnd_x[1], rnd_y[0] : rnd_y[1]] = 1
        tile_replacement = torch.ones(crop.shape)
        crop = torchvision.transforms.Resize(((crop.shape[1] // 3), crop.shape[2]))(crop)

        tile_replacement[:, : crop.shape[1], :] = crop[:, :, :]

        data[:, rnd_x[0] : rnd_x[1], rnd_y[0] : rnd_y[1]] = tile_replacement[:, :, :]
        return data.float()


class RandomArtifactAugmentation(Transform):
    def __init__(self, probability=0.7) -> None:
        self.artifact_augs = [
            # RandomTearAugmentation(),
            RandomLineAugmentation(),
            RandomBlobAugmentation(),
        ]
        self.p = probability

    def __call__(self, data: Dict[str, Any]):
        rnd = torch.rand(len(self.artifact_augs))
        for i in range(len(self.artifact_augs)):
            if rnd[i] < self.p:
                data["image"] = self.artifact_augs[i](data["image"])
        data["image"] = data["image"].float()
        return data
