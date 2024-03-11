from enum import Enum


class ImageVariants(Enum):
    RGB = 'rgb'
    YCbCr = 'ycbcr'

    def __str__(self):
        return self.value

    @staticmethod
    def get_variant_channels(image_variant):
        if image_variant == ImageVariants.RGB:
            return 3
        elif image_variant == ImageVariants.YCbCr:
            return 1
        else:
            raise ValueError(f"Unknown image variant {image_variant}")
    