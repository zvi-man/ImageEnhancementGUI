from backend import AugmentationPipe, AugmentationMethod, AugmentationUtils


def init_aug_pipe() -> AugmentationPipe:
    reshape = AugmentationMethod(name="Reshape",
                                 func=AugmentationUtils.reshape,
                                 func_args={"width": 256,
                                            "height": 256})
    blurring = AugmentationMethod(name="Blurring",
                                  func=AugmentationUtils.blur,
                                  func_args={"radius": 1})
    mirror = AugmentationMethod(name="Mirror",
                                func=AugmentationUtils.mirror,
                                func_args={})
    subsample = AugmentationMethod(name="Subsample",
                                   func=AugmentationUtils.subsample,
                                   func_args={"resize_factor": 0.5})
    sharpening = AugmentationMethod(name="Sharpening",
                                    func=AugmentationUtils.sharpening,
                                    func_args={"radius": 2})
    brightness = AugmentationMethod(name="Brightness",
                                    func=AugmentationUtils.brightness,
                                    func_args={"brightness_factor": 1.0})
    zoom = AugmentationMethod(name="zoom",
                              func=AugmentationUtils.zoom,
                              func_args={"top_factor": 0.0,
                                         "bot_factor": 0.0,
                                         "left_factor": 0.0,
                                         "right_factor": 0.0})
    motion = AugmentationMethod(name="Motion",
                                func=AugmentationUtils.motion,
                                func_args={"radius": 5})
    cj = AugmentationMethod(name="ColorJitter",
                            func=AugmentationUtils.color_jitter,
                            func_args={"brightness": 0.3,
                                       "contrast": 0.3,
                                       "saturation": 0.1,
                                       "hue": 0.0})
    ra = AugmentationMethod(name="RandomAffine",
                            func=AugmentationUtils.random_affine,
                            func_args={"degrees": 10,
                                       "scale": 0.1,
                                       "shear": 0.1})
    return AugmentationPipe([reshape, sharpening, blurring, mirror, subsample, brightness,
                             zoom, motion, cj, ra])
