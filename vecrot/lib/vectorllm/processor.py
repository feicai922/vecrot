"""图像 + 文本 → 模型输入的 processor（`<pixel>` 在这里展开成 64 个视觉占位 token）。

v1 在模块 import 时就往 HF 的 Auto* 映射里塞了自定义类，这在 transformers 5.x 下会崩
（见 `register_for_auto_classes` 的注释）。v2 把注册改成显式可选。
"""

import numpy as np
from transformers import AutoImageProcessor, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from .image_processor import VectorLLMImageProcessor


class VectorLLMImagesKwargs(ImagesKwargs):
    resized_size: int
    patch_size: int


class VectorLLMProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: VectorLLMImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        }
    }


class VectorLLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "VectorLLMImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<pixel>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        # 换 LLM 后 tokenizer 可能是 Llama/Gemma/GPT2 系列；ProcessorMixin 会按
        # `tokenizer_class` 校验实际类型，这里把允许集合放宽成"当前传入的类型"。
        self.tokenizer_class = (type(tokenizer).__name__,)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[VectorLLMProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            VectorLLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if images is not None:
            num_image_tokens = (
                self.image_processor.resized_size // self.image_processor.patch_size
            ) ** 2
            for index in range(len(text)):
                while self.image_token in text[index]:
                    text[index] = text[index].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                text[index] = text[index].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = VectorLLMProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            resized_size = images_kwargs.get("resized_size", None) or self.image_processor.resized_size
            patch_size = images_kwargs.get("patch_size", None) or self.image_processor.patch_size
            num_image_patches = [(resized_size // patch_size) ** 2 for _ in image_sizes]
            vision_data.update(
                {"num_image_tokens": num_image_patches, "num_image_patches": num_image_patches}
            )
        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


def register_for_auto_classes() -> None:
    """把自定义 processor 注册进 HF 的 Auto* 映射（只有想用 `AutoProcessor.from_pretrained`
    加载"带代码的权重目录"时才需要调用）。

    v1 在模块顶层无条件做了这两次注册，其中
        `AutoImageProcessor.register("VectorLLMImageProcessor", cls)`
    在 transformers 5.x 里会被解释成 `(config_class, slow_image_processor_class)`，
    映射值变成 `{"pil": cls}`，随后 `AutoProcessor.from_pretrained` 在
    `get_possibly_dynamic_module` 里崩 `AttributeError: 'dict' object has no attribute '__name__'`。
    v2 默认**不注册**（我们直接构造 processor），只在显式调用本函数时才注册，
    并且顺带把映射值修正成类对象本身，兼容 4.x / 5.x。
    """
    import transformers

    try:
        AutoProcessor.register("VectorLLMProcessor", VectorLLMProcessor)
    except ValueError:
        pass
    try:
        AutoImageProcessor.register("VectorLLMImageProcessor", VectorLLMImageProcessor)
    except Exception:
        pass
    mapping = getattr(transformers, "IMAGE_PROCESSOR_MAPPING", None)
    if mapping is not None:
        mapping._extra_content["VectorLLMImageProcessor"] = VectorLLMImageProcessor


__all__ = [
    "VectorLLMImagesKwargs",
    "VectorLLMProcessorKwargs",
    "VectorLLMProcessor",
    "register_for_auto_classes",
]
