from typing import Tuple
import streamlit as st
from PIL import Image

from gui_config import init_aug_pipe
from backend import AugMode

# Constants
NUM_OF_COL = 5
IMAGE_TYPES = ["png", "jpg", "jpeg"]
DEFAULT_NUM_OF_IMAGES = 10
NUM_IM_STEP = 10
MIN_NUM_IMAGES = 0
NUM_IMAGES_ROW = 5


def add_centered_title(title: str) -> None:
    st.markdown(f"<h1 style='text-align: center;'>{title}</h1>", unsafe_allow_html=True)


def add_centered_text(text: str) -> None:
    st.markdown(f"<p style='text-align: center;'><strong>{text}</strong></p>", unsafe_allow_html=True)


class DataAugmentationGUI(object):
    def __init__(self) -> None:
        self.init_session_state()
        self.init_sidebar()
        self.init_main_window()

    @staticmethod
    def init_session_state() -> None:
        if 'augmentation_pipe' not in st.session_state:
            st.session_state.augmentation_pipe = init_aug_pipe()
            for aug_method in st.session_state.augmentation_pipe.augmentation_list:
                if f"{aug_method.name}, AugMode" not in st.session_state:
                    st.session_state[f"{aug_method.name}, AugMode"] = aug_method.aug_mode.name
                if f"prob {aug_method.name} default val" not in st.session_state:
                    st.session_state[f"prob {aug_method.name} default val"] = aug_method.use_aug_at_probability
                for arg_name, arg_val in aug_method.func_args.items():
                    if f"{aug_method.name}, {arg_name} default val" not in st.session_state:
                        st.session_state[f"{aug_method.name}, {arg_name} default val"] = arg_val
                for arg_name, arg_std in aug_method.func_args_std.items():
                    if f"{aug_method.name}, {arg_name} std default val" not in st.session_state:
                        st.session_state[f"{aug_method.name}, {arg_name} std default val"] = arg_std

    @staticmethod
    def init_sidebar() -> None:
        with st.sidebar:
            st.title("Select Image Enhancement")
            for aug_method in st.session_state.augmentation_pipe.augmentation_list:
                st.subheader(aug_method.name)
                st.radio(
                    f"Select {aug_method.name} Options",
                    (AugMode.NotActive.name, AugMode.Active.name),
                    key=f"{aug_method.name}, AugMode",
                    horizontal=True
                )
                aug_method.aug_mode = AugMode[st.session_state[f"{aug_method.name}, AugMode"]]
                if aug_method.aug_mode != AugMode.NotActive:
                    st.text("Select Augmentation Value")
                    for arg_name, arg_val in aug_method.func_args.items():
                        step = 1 if aug_method.func_arg_type[arg_name] == int else 0.1
                        min_value = 0 if aug_method.func_arg_type[arg_name] == int else 0.0
                        default_val = st.session_state[f"{aug_method.name}, {arg_name} default val"]
                        new_func_arg_val = st.number_input(arg_name, value=default_val, min_value=min_value,
                                                           step=step, key=f"{aug_method.name}, {arg_name}")
                        # Make sure the given value is of the correct class
                        new_func_arg_val = aug_method.func_arg_type[arg_name](new_func_arg_val)
                        aug_method.func_args[arg_name] = new_func_arg_val
                st.write("##")

    def init_main_window(self) -> None:
        window = st.container()
        with window:
            add_centered_title("Image Enhancement GUI")
            st.write("##")
            original_image_path = st.file_uploader("Upload original image", type=IMAGE_TYPES,
                                                   accept_multiple_files=False)
            # st.write(st.session_state.augmentation_pipe)
            if original_image_path is not None:
                input_im = Image.open(original_image_path)
                self.display_original_image(input_im, original_image_path.name)
                st.write("##")
                self.display_augmented_image(input_im)

    @staticmethod
    def display_original_image(input_im: Image.Image, image_name: str) -> Image.Image:
        add_centered_text(f"Original Image: {image_name}")
        st.image(input_im, use_column_width=True)
        return input_im

    def display_augmented_image(self, input_im: Image.Image) -> None:
        add_centered_text(f"Image After Enhancement")
        aug_im, im_name = self.augment_main_image(input_im)
        st.image(aug_im, use_column_width=True, caption=im_name)

    @staticmethod
    def augment_main_image(input_im: Image.Image) -> Tuple[Image.Image, str]:
        aug_im, im_name = st.session_state.augmentation_pipe.augment_image(input_im, random=False)
        return aug_im, im_name


DataAugmentationGUI()
