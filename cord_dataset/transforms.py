from typing import Union, Tuple, Optional, Any, Iterable, List

import torch
from PIL.Image import Image
from torch import Tensor
import torchvision.transforms.functional as F

from .objects import Attributes, BoundingBox
from .utils import bbox_frac_to_pixels, is_pixel_bbox


class TransformOutput:
    def __init__(self, img, attributes, params=None):
        self.img: Union[Image, Tensor] = img
        self.attributes: Attributes = attributes
        self.params: Optional[Any] = params

    # Allow unpacking of the attributes
    def __iter__(self):
        return iter((self.img, self.attributes, self.params))


class Transform:
    def forward(
        self, img: Union[Image, Tensor], attributes: Attributes
    ) -> Tuple[Union[Image, Tensor], Attributes, Optional[Any]]:
        pass

    def __call__(
        self, img: Union[Image, Tensor], attributes: Attributes
    ) -> TransformOutput:
        """
        Args:
            img: The image can be either PIL image, or it may be a torch tensor if
                ToTensor transform has been applied.
            attributes: Attributes object containing (bbox | polygon) and classes. Make
                sure to also modify polygon or bounding box to transform, when necessary.

        Returns:
            img, attributes, Optional[out_params]
        """
        img, attributes, *params = self.forward(img, attributes)
        return TransformOutput(img, attributes, params)


class Compose(Transform):
    def __init__(self, transforms: Iterable):
        self.transforms = transforms

    def forward(
        self, img: Union[Image, Tensor], attributes: Attributes
    ) -> TransformOutput:
        """
        Compose multiple transforms by applying each transform in list to the input sequentially.
        :param img: The image to be transformed
        :param attributes: The attributes (classes, bbox, polygon)
        :return: The transformed output.
        """
        out_params = []
        for T in self.transforms:
            img, attributes, *params = T(img, attributes)
            out_params.append(params)
        return TransformOutput(img, attributes, out_params)


class ToTensor(Transform):
    def forward(self, img: Image, attributes: Attributes):
        if isinstance(img, torch.Tensor):
            return img, attributes

        return F.to_tensor(img), attributes


class ToPixelCoordinates(Transform):
    def forward(self, img: Union[torch.Tensor, Image], attributes: Attributes):
        if isinstance(img, Image):
            w, h = img.size
        else:
            h, w = img.shape[-2:]

        attributes.bbox = bbox_frac_to_pixels(attributes.bbox, img_h=h, img_w=w)
        return img, attributes


class ResizedCropToBoundingBox(Transform):
    def __init__(
        self,
        size: List[int],
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor, attributes: Attributes):
        assert (
            attributes.bbox is not None
        ), "(Image, Attributes) pair does not contain bounding box"
        assert is_pixel_bbox(
            attributes.bbox
        ), "Use the ToPixelCoordinats transform before applying CropTransform"

        b = attributes.bbox
        img = F.resized_crop(
            img,
            top=b.y,
            left=b.x,
            height=b.h,
            width=b.w,
            size=self.size,
            interpolation=self.interpolation,
        )
        attributes.bbox = BoundingBox(x=0, y=0, h=self.size[0], w=self.size[1])

        return img, attributes


class RandomHorizontalFlip(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def forward(self, img: Union[torch.Tensor, Image], attributes: Attributes):
        if torch.rand(1) < self.p:
            img = F.hflip(img)
        return img, attributes


class RandomCrop(Transform):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: [torch.Tensor], attributes: Attributes):
        h, w = image.shape[-2:]
        new_h, new_w = self.output_size

        top = torch.randint(h - new_h, size=(1,)).item()
        left = torch.randint(w - new_w, size=(1,)).item()

        image = image[..., top : top + new_h, left : left + new_w]
        if attributes.polygon is not None:
            attributes.polygon -= [left, top]

        if attributes.bbox is not None:
            b = attributes.bbox
            attributes.bbox = BoundingBox(
                x=b.x - left, y=b.y - top, h=min(b.h, new_h), w=min(b.w, new_w)
            )

        return image, attributes, (top, left, new_h, new_w)
