"""
Geojson is a common format for storing the annotations on gigapixel 2D images.

Our GeoJSON utility helps with two use-cases:

## Region annotations

Extracting windows from annotated regions, often used in multiclass classification.
A single annotations is usually larger than a window.

## Object annotations
 
Extracting windows with annotated objects, often used in detection and segmentation.
A single annotation is usually smaller than a window and enables multi-level window extraction.

"""

from .GeoAnno import GeoAnno, AnnotationType
from .AnnoFilterConfig import AnnoFilterConfig
from .GeojsonRegionWindows import GeojsonRegionWindows
from .GeojsonObjWindows import GeojsonObjWindows
from .NoUsefulWindowException import NoUsefulWindowException
from .WSIGeojsonDataset import WSIGeojsonDataset
