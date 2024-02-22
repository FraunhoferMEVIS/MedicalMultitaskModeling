import random
from typing import Literal, Iterable
from pydantic import Field
import itertools

from shapely.geometry import Polygon
from shapely import GEOSException

from mmm.BaseModel import BaseModel

from .GeoAnno import GeoAnno
from .NoUsefulWindowException import NoUsefulWindowException


class GeojsonRegionWindows(BaseModel):
    """
    Yields subwindows inside annotated regions given as coarse polygons.
    """

    min_area: float = Field(
        default=0.95,
        description="Area of a window that is required to be inside the annotated region. 0.95 for 95%",
    )
    patch_size: tuple[int, int] = 224, 224
    coordinate_augmentation: Literal["random", "none"] = "random"
    windowsize_augmentation: Literal["relative", "none"] = "relative"
    augmentation_strength: float = Field(
        default=0.2,
        description="Controls the windowsize augmentation. Should be smaller than 1.",
    )
    stepsize: int = Field(
        default=2,
        description="Controls the stepsize. For one moves one window at a time, for 2 it moves half a window every step",
    )

    def augment_window(self, x_raw, y_raw, l0_window_width, l0_window_height):
        """
        If corresponding config options are set, will jiggle coordinates and window size.
        """
        if self.coordinate_augmentation == "random":
            rand_x = int(l0_window_width // (self.stepsize * 2))
            rand_y = int(l0_window_height // (self.stepsize * 2))
            x = x_raw + random.randint(-1 * rand_x, rand_x)
            y = y_raw + random.randint(-1 * rand_y, rand_y)
        elif self.coordinate_augmentation == "none":
            x, y = x_raw, y_raw
        else:
            raise NotImplementedError(f"Unknown stepsize_method {self.coordinate_augmentation}")

        if self.windowsize_augmentation == "relative":
            rand_width = int(l0_window_width * self.augmentation_strength)
            rand_height = int(l0_window_height * self.augmentation_strength)
            width = l0_window_width + random.randint(-1 * rand_width, rand_width)
            height = l0_window_height + random.randint(-1 * rand_height, rand_height)
        elif self.windowsize_augmentation == "none":
            width, height = l0_window_width, l0_window_height
        else:
            raise NotImplementedError(f"Unknown windowsize augmentation method {self.windowsize_augmentation}")

        return Polygon([(x, y), (x + width, y), (x + width, y + height), (x, y + height)])

    def test_window_validity(self, checkwindow: Polygon, anno: GeoAnno) -> bool:
        try:
            if checkwindow.intersects(anno.shape):
                if (checkwindow.intersection(anno.shape).area / checkwindow.area) >= self.min_area:
                    return True
        except GEOSException:
            return False
        return False

    def iter_valid_windows(
        self, anno: GeoAnno, level_downsamples: dict[int, float]
    ) -> Iterable[tuple[int, tuple[int, int], tuple[int, int]]]:
        """
        Yields all windows with sufficient area inside `anno`.

        Uses a generator. As a result you can e.g. use `itertools.islice` to limit the number of windows per annotation.

        The patch_size is given. In consequence, only (level, location) tuples are yielded.
        """
        window_sizes = {
            l: (
                self.patch_size[0] * downsample_fac,
                self.patch_size[1] * downsample_fac,
            )
            for l, downsample_fac in level_downsamples.items()
        }

        # It does not make sense if a window is larger than the region annotation itself
        useful_window_sizes = {k: v for k, v in window_sizes.items() if v[0] * v[1] <= anno.shape.area}
        if not useful_window_sizes:
            raise NoUsefulWindowException(
                f"Annotation did not have a useful window in levels {level_downsamples}: {anno}"
            )
        proposal_coords = {}
        for l, (l0_window_width, l0_window_height) in useful_window_sizes.items():
            xcoords = list(
                range(
                    int(anno.shape.bounds[0] - l0_window_width),
                    int(anno.shape.bounds[2]),
                    int(l0_window_width // self.stepsize),
                )
            )
            ycoords = list(
                range(
                    int(anno.shape.bounds[1] - l0_window_height),
                    int(anno.shape.bounds[3]),
                    int(l0_window_height // self.stepsize),
                )
            )

            # Itertools.product does not yield random combinations, so a list conversion is necessary
            cs = list(itertools.product(xcoords, ycoords))
            random.shuffle(cs)
            proposal_coords[l] = (coord for coord in cs)

        valid_levels: list[int] = list(useful_window_sizes.keys())
        # current_level_idx = 0  # in first iteration it will be 0
        while valid_levels:
            selected_level = random.choice(valid_levels)
            try:
                # selected_level = valid_levels[current_level_idx]
                x_raw, y_raw = next(proposal_coords[selected_level])
                l0_window_width, l0_window_height = useful_window_sizes[selected_level]
                l0_window: Polygon = self.augment_window(x_raw, y_raw, l0_window_width, l0_window_height)
                if self.test_window_validity(l0_window, anno):
                    x1, y1, x2, y2 = l0_window.bounds
                    level_patchsize = (
                        int((x2 - x1) / level_downsamples[selected_level]),
                        int((y2 - y1) / level_downsamples[selected_level]),
                    )
                    yield selected_level, (int(x1), int(y1)), level_patchsize
                    # current_level_idx = (current_level_idx + 1) % len(valid_levels)
            except StopIteration:
                # No more proposal coordinates in current_level
                valid_levels.remove(selected_level)
                # current_level_idx = 0
