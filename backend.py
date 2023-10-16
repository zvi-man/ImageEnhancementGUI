from dataclasses import dataclass, field
from typing import Callable, List, Any, Dict, Tuple
from enum import IntEnum
import torchvision.transforms as T

import PIL
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2  # type: ignore

# Constants
DELIMITER = "_"


class AugMode(IntEnum):
    NotActive = 0
    Active = 1


def get_true_on_probability(probability: float) -> bool:
    if probability < 0 or probability > 1:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0:
        return False
    # noinspection PyArgumentList
    return np.random.random_sample() <= probability


@dataclass
class AugmentationMethod:
    name: str
    func: Callable
    func_args: Dict[str, Any]
    func_args_std: Dict[str, Any] = field(default_factory=dict)
    aug_mode: AugMode = AugMode.NotActive
    use_aug_at_probability: float = 0.5
    func_arg_type: Dict[str, type] = field(init=False)

    def __post_init__(self) -> None:
        self.func_arg_type = dict()
        for arg_name, arg_val in self.func_args.items():
            self.func_arg_type[arg_name] = type(arg_val)
        if not self.func_args_std:
            for arg_name in self.func_args:
                self.func_args_std[arg_name] = self.func_arg_type[arg_name](0)

    def augment_image_no_random(self, pil_im: Image.Image) -> Tuple[Image.Image, str]:
        if self.aug_mode == AugMode.NotActive:
            return pil_im, ""
        func_arg_values = [val for key, val in self.func_args.items()]
        report_str = self.gen_report_str(func_arg_values)
        return self.func(pil_im, **self.func_args), report_str

    def augment_image_with_random(self, pil_im: Image.Image) -> Tuple[Image.Image, str]:
        if self.aug_mode != AugMode.NotActive:
            if self.aug_mode == AugMode.Active:
                return self.augment_image_no_random(pil_im)
            # self.aug_mode == AugMode.Random
            if get_true_on_probability(self.use_aug_at_probability):
                func_args_randomized = self.get_func_args_randomized()
                func_arg_values = [val for key, val in func_args_randomized.items()]
                report_str = self.gen_report_str(func_arg_values)
                return self.func(pil_im, **func_args_randomized), report_str
        return pil_im, ""

    def get_func_args_randomized(self) -> Dict[str, Any]:
        func_args_randomized = self.func_args.copy()
        for arg_name, arg_val in func_args_randomized.items():
            if isinstance(arg_val, int):
                rand_val = np.random.randint(-self.func_args_std[arg_name], self.func_args_std[arg_name] + 1)
            else:
                rand_val = np.random.uniform(-self.func_args_std[arg_name], self.func_args_std[arg_name])
            func_args_randomized[arg_name] = arg_val + rand_val
        return func_args_randomized

    def gen_report_str(self, argument_values: List[Any]) -> str:
        args_str = DELIMITER.join(map(self.int_or_float_to_str, argument_values))
        return self.name + DELIMITER + args_str

    @staticmethod
    def int_or_float_to_str(x: Any) -> str:
        return str(x) if isinstance(x, int) else f"{x:.3}"


@dataclass
class AugmentationPipe:
    augmentation_list: List[AugmentationMethod]

    def augment_image(self, pil_im: Image.Image, random: bool = False) -> Tuple[Image.Image, str]:
        image_name = ''
        for aug_method in self.augmentation_list:
            if random:
                pil_im, aug_str = aug_method.augment_image_with_random(pil_im)
            else:
                pil_im, aug_str = aug_method.augment_image_no_random(pil_im)
            if aug_str:
                image_name += aug_str + "_"
        return pil_im, image_name


class AugmentationUtils:

    @staticmethod
    def reshape(input_im: Image.Image, width: int, height: int) -> Image.Image:
        return input_im.resize([width, height])

    @staticmethod
    def blur(input_im: Image.Image, radius: int) -> Image.Image:
        gaussian_filter = ImageFilter.GaussianBlur(radius=radius)
        output_im = input_im.filter(gaussian_filter)
        return output_im

    @staticmethod
    def mirror(input_im: Image.Image) -> Image.Image:
        return ImageOps.mirror(input_im)

    @staticmethod
    def subsample(input_im: Image.Image, resize_factor: float,
                  return_original_size: bool = True) -> Image.Image:
        original_size = input_im.size
        new_size = round(original_size[0] * resize_factor), round(original_size[1] * resize_factor)
        output_im = input_im.resize(new_size)
        if return_original_size:
            output_im = output_im.resize(original_size)
        return output_im

    @staticmethod
    def sharpening(input_im: Image.Image, radius: int) -> Image.Image:
        sharpening_filter = ImageFilter.UnsharpMask(radius=radius)
        output_im = input_im.filter(sharpening_filter)
        return output_im

    @staticmethod
    def motion(input_im: Image.Image, radius: int) -> Image.Image:
        kernel_motion_blur = np.zeros((radius, radius))
        kernel_motion_blur[int((radius - 1) / 2), :] = np.ones(radius)
        kernel_motion_blur = kernel_motion_blur / radius
        input_np_im = np.array(input_im)
        # noinspection PyUnresolvedReferences
        output_np_im = cv2.filter2D(input_np_im, -1, kernel_motion_blur)
        return PIL.Image.fromarray(output_np_im)

    @staticmethod
    def zoom(input_im: Image.Image, top_factor: float,
             bot_factor: float, left_factor: float, right_factor: float) -> Image.Image:
        original_size = input_im.size
        width, height = original_size
        top_pix = round(top_factor * height)
        bot_pix = height - round(bot_factor * height)
        left_pix = round(left_factor * height)
        right_pix = width - round(right_factor * height)
        cropped_im = input_im.crop((left_pix, top_pix, right_pix, bot_pix))
        return cropped_im.resize(original_size)

    @staticmethod
    def brightness(input_im: Image.Image, brightness_factor: float) -> Image.Image:
        """
        brightness_factor == 1 - image same
        brightness_factor = 0.5 - darkens the image
        brightness_factor = 1.5 - brightens the image
        """
        enhancer = ImageEnhance.Brightness(input_im)
        output_im = enhancer.enhance(brightness_factor)
        return output_im

    @staticmethod
    def color_jitter(input_im: Image.Image, brightness: float, contrast: float,
                     saturation: float, hue: float) -> Image.Image:
        jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        return jitter(input_im)

    @staticmethod
    def random_affine(input_im: Image.Image, degrees: int = 10,
                      scale: float = 0.1, shear: float = 0.1) -> Image.Image:
        affine = T.RandomAffine(degrees=degrees, translate=None, scale=[1 - scale, 1 + scale],
                                shear=shear, resample=False, fillcolor=0)
        return affine(input_im)
