import random
from urllib.parse import urljoin
import pickle
import os
from pathlib import Path
import time
import requests
import numpy as np
from typing import Dict, Optional, Any, OrderedDict, Tuple, Literal
import imageio.v3 as imageio
from urllib3.exceptions import InsecureRequestWarning
from mmm.utils import get_default_cachepath, unique_str_hash

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)  # type: ignore


# @disk_cacher()
def get_image_as_numpy(url: str, attempts=5) -> np.ndarray:
    res = requests.get(url, verify=os.getenv("NOMAD_CACERT", default=False))
    b = res.content
    for attempt in range(attempts):
        try:
            return imageio.imread(b).copy()
        except Exception as e:
            print(e)
            e1 = f"Error opening {url} as numpy image in attempt {attempt}"
            print(e1)
            e2 = f"Cannot convert {res.text} to an image with the following exception:"
            print(e2)
            time.sleep(np.random.randint(2, 59))

    # dump_f = os.getenv("ML_DATA_OUTPUT", default="/data_output/") + "wsi-service-dump.pkl"
    # with open(dump_f, 'wb') as f:
    #     torch.save(res.content, f)
    # logging.info(f"Dumped {res.text}")
    raise Exception(f"{attempts} failed attempts to open Image {url}")


async def get_image_as_numpy_async(url: str, session=None) -> np.ndarray:
    """
    Calls an http and returns the result as numpy.

    For example, for the full url to an endpoint `full_endpoint_url` outside of async you can use:

    ```python
    return_value: np.ndarray = await get_rest_endpoint_as_numpy(full_endpoint_url)
    ```
    """
    if session is None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            resp = await session.get(url)
    else:
        resp = await session.get(url)
    b = await resp.read()
    try:
        return imageio.imread(b).copy()
    except Exception as e:
        print(e)
        e1 = f"Error opening {url} as numpy image"
        print(e1)
        resp_text = await resp.text()
        e2 = f"Cannot convert {resp_text} to an image with the following exception:"
        print(e2)
        raise Exception(f"{e}\n{e1}\n{e1}")


def get_json(url: str) -> Any:
    return requests.get(url, verify=os.getenv("NOMAD_CACERT", default=False)).json()


class StorageMapper:
    def __init__(
        self,
        base_url: str,
        storage_path: Path,
        slideid_strategy: Literal["absolutepath", "filename"],
    ) -> None:
        """
        The storage path is necessary to know at which point to cut the paths to the slides.
        """
        self.base_url = base_url
        self.storage_path: Path = storage_path
        self.id_strategy = slideid_strategy

    def build_request_body(self, slide_id: str, absolute_slide_path: Path) -> dict:
        return {
            "slide_id": f"{slide_id}",
            "storage_type": "fs",
            "storage_addresses": [
                {
                    "address": str(absolute_slide_path.relative_to(self.storage_path)),
                    "slide_id": f"{slide_id}",
                    "main_address": True,
                    "storage_address_id": f"{slide_id}",
                }
            ],
        }

    def get_info(self, slide_id: str) -> dict:
        response = requests.get(f"{self.base_url}v3/slides/{slide_id}/")
        return response.json()

    def upload_slide(self, slide_id: str, slide_path: Path) -> str:
        request_body = self.build_request_body(slide_id, slide_path)
        response = requests.post(f"{self.base_url}v3/slides/", json=request_body)
        return response.json()

    def get_unique_slideid(self, absolute_slide_path: Path) -> str:
        if self.id_strategy == "absolutepath":
            relative_path = absolute_slide_path.relative_to(self.storage_path)
            return str(relative_path).replace("/", "_").replace(".", "_")
        elif self.id_strategy == "filename":
            return absolute_slide_path.name.replace(".", "_")


class WSIService:
    """
    Wrapper around one instance of a running WSI-service.

    Just instantiating the object will already make calls to the WSI-service for basic indexing stuff.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        # Memorizes detailed info for each slide
        self._slideinfo_memoization: Dict[str, Any] = {}

    def get_is_alive_route(self) -> str:
        return urljoin(self.base_url, "alive")

    def get_slide_info(self, slide_id: str) -> Dict:
        if slide_id not in self._slideinfo_memoization:
            u = f"{self.base_url}v3/slides/{slide_id}/info?plugin=tiffslide"
            d = get_json(u)
            self._slideinfo_memoization[slide_id] = d

        return self._slideinfo_memoization[slide_id]

    def build_link_for_region(
        self,
        wsi_service_img_id: str,
        level=2,
        position: Optional[Tuple[int, ...]] = None,
        spatial_size: Optional[Tuple[int, ...]] = None,
        format="png",
        pos_level_0=False,
    ) -> str:
        u = f"{self.base_url}v3/slides/{wsi_service_img_id}/region/level/{level}"

        if position is None:
            position = (0, 0)
        if pos_level_0:
            position = (
                int(position[0] / self.get_slide_info(wsi_service_img_id)["levels"][level]["downsample_factor"]),
                int(position[1] / self.get_slide_info(wsi_service_img_id)["levels"][level]["downsample_factor"]),
            )
        u += f"/start/{'/'.join(map(str, position))}"

        if spatial_size is None:
            spatial_size = (
                self.get_slide_info(wsi_service_img_id)["levels"][level]["extent"]["x"] - position[0],
                self.get_slide_info(wsi_service_img_id)["levels"][level]["extent"]["y"] - position[1],
            )

        u += f"/size/{'/'.join(map(str, spatial_size))}"
        u += f"?image_format={format}"
        u += f"&plugin=tiffslide"
        return u


class LocalServiceMapper:
    """
    Uses an instance of WSI service and a StorageMapper to map local paths to WSI-service ids implicitly.
    If the mounts are correct this should feel local.

    The mounts are correct if both the WSI-service and the user have access to the same volume,
    which should be mounted into /data of the WSI-service.
    The storage mapper does not need access.

    It needs to memorize which slides are already indexed in the storage mapper.
    """

    def __init__(
        self,
        wsi_service_base: str,
        storage_mapper_base: str,
        storage_mapper_root: Path,
        slideid_strategy: Literal["absolutepath", "filename"],
    ) -> None:
        unique_filename = unique_str_hash(wsi_service_base, storage_mapper_base, storage_mapper_root)
        self.cache_file_path = get_default_cachepath() / unique_filename
        self.wsi_service = WSIService(wsi_service_base)
        self.storage_mapper = StorageMapper(storage_mapper_base, storage_mapper_root, slideid_strategy)

        if not self.cache_file_path.exists():
            # Create file
            self.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file_path, "wb") as f:
                # Save using pickle
                pickle.dump({}, f)
        self._load_from_disk()

    def _load_from_disk(self):
        # Load cache file
        with open(self.cache_file_path, "rb") as f:
            self.slides: Dict[str, Path] = pickle.load(f)

    def get_slide_id_by_path(self, absolute_slide_path: Path) -> str:
        return self.storage_mapper.get_unique_slideid(absolute_slide_path)

    def index_wsis(self, relative_root, forget_old=False, wsi_glob: str = "**/*.svs"):
        if forget_old:
            self.slides = {}

        root = self.storage_mapper.storage_path / relative_root
        for i, wsi_path in enumerate(root.glob(wsi_glob)):
            slide_id = self.storage_mapper.get_unique_slideid(wsi_path)
            assert (not forget_old) or slide_id not in self.slides, f"Slide {slide_id} already indexed"
            if slide_id not in self.slides:
                self.storage_mapper.upload_slide(slide_id, wsi_path)
                self.slides[slide_id] = wsi_path

        # Commit the already indexed slide ids to the cache file
        with open(self.cache_file_path, "wb") as f:
            pickle.dump(self.slides, f)

    def get_viewer_url(self, slide_id: str) -> str:
        return urljoin(self.wsi_service.base_url, f"v3/slides/{slide_id}/viewer")

    def get_all_ids(self, implicitly_update=True) -> list[str]:
        if implicitly_update:
            self._load_from_disk()
        return list(self.slides.keys())

    def __repr__(self) -> str:
        return f"""
        LocalServiceMapper with {len(self.slides)} slides indexed.
        WSI-service address: {self.wsi_service.base_url}v3/docs/
        Storage mapper address: {self.storage_mapper.base_url}v3/docs
        """

    def select_random_window(self, slide_id: str, tile_size: int = 1024) -> tuple[str, tuple[int, int, int, int]]:
        """
        For the slide, returns a wsi-service link to a random window of size tile_size x tile_size.
        The second value is the region of interest.
        """
        img_info = self.wsi_service.get_slide_info(slide_id)
        # if "levels" not in img_info:
        #     raise Exception(f"Slide {slide_id} has no levels")
        num_levels = len(img_info["levels"])
        # Select a level, the higher the level the less probable it should be picked
        random_level = random.choices(range(num_levels), weights=[2**i for i in reversed(range(num_levels))])[0]
        downsample_fac = img_info["levels"][random_level]["downsample_factor"]
        dims = (
            img_info["levels"][random_level]["extent"]["x"],
            img_info["levels"][random_level]["extent"]["y"],
        )
        pos = random.randint(0, max(0, dims[0] - tile_size)), random.randint(0, max(0, dims[1] - tile_size))
        image_url = self.wsi_service.build_link_for_region(
            wsi_service_img_id=slide_id,
            level=random_level,
            position=pos,
            spatial_size=(tile_size, tile_size),
        )

        # Build HTML snippet for viewer
        roi_x0 = int(pos[0] * downsample_fac)
        roi_x1 = int(pos[0] * downsample_fac + (tile_size * downsample_fac))
        # openlayers has an inverted y axis
        roi_y0 = img_info["extent"]["y"] - int(pos[1] * downsample_fac)
        roi_y1 = img_info["extent"]["y"] - int(pos[1] * downsample_fac + (tile_size * downsample_fac))

        return image_url, (roi_x0, roi_y0, roi_x1, roi_y1), random_level
