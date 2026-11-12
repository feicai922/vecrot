"""VectorLLM 的图像预处理。

⚠️ 两段 resize（432 → 128）**不是笔误**：训练侧的 xtuner pipeline 是先把图像
resize 到 432（`pre_resize_size`）再进 image_processor 缩到 `resized_size`=128
（见 `tools/convert_from_xtuner.py::run_xtuner_reference`）。这里刻意复刻该流程，
改成"一步 resize 到 128"会与训练/对齐口径产生偏差。要动它请先跑 `tests/test_parity_v1.py`。

默认：BICUBIC、rescale 1/255、**不做 mean/std 归一化**（`do_normalize=False`）。
"""

from typing import Optional, Union

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
)
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class VectorLLMImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = False,
        image_mean=None,
        image_std=None,
        do_convert_rgb: bool = True,
        pre_resize_size: Optional[int] = 432,
        resized_size: int = 128,
        patch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pre_resize_size = pre_resize_size
        self.resized_size = resized_size
        self.patch_size = patch_size
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean=None,
        image_std=None,
        pre_resize_size: Optional[int] = None,
        resized_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_flat_list_of_images(images)
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "Input images already look rescaled. Set do_rescale=False to avoid double rescaling."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        processed_images = []
        for image in images:
            if do_resize:
                if pre_resize_size is not None:
                    image = resize(
                        image,
                        size=(pre_resize_size, pre_resize_size),
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                image = resize(
                    image,
                    size=(resized_size, resized_size),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(
                image,
                data_format,
                input_channel_dim=input_data_format,
            )
            processed_images.append(image)

        return processed_images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean=None,
        image_std=None,
        pre_resize_size: Optional[int] = None,
        resized_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        do_resize = self.do_resize if do_resize is None else do_resize
        resample = self.resample if resample is None else resample
        do_rescale = self.do_rescale if do_rescale is None else do_rescale
        rescale_factor = self.rescale_factor if rescale_factor is None else rescale_factor
        do_normalize = self.do_normalize if do_normalize is None else do_normalize
        image_mean = self.image_mean if image_mean is None else image_mean
        image_std = self.image_std if image_std is None else image_std
        pre_resize_size = self.pre_resize_size if pre_resize_size is None else pre_resize_size
        resized_size = self.resized_size if resized_size is None else resized_size
        do_convert_rgb = self.do_convert_rgb if do_convert_rgb is None else do_convert_rgb

        images = self._preprocess(
            images=images,
            do_resize=do_resize,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            pre_resize_size=pre_resize_size,
            resized_size=resized_size,
            do_convert_rgb=do_convert_rgb,
            data_format=data_format,
            input_data_format=input_data_format,
        )
        return BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)
