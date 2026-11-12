import copy

from transformers import PretrainedConfig, Qwen3Config

from .radio.hf_model import RADIOConfig as HFRADIOConfig


class ProjectorConfig(PretrainedConfig):
    model_type = "vectorllm_0407_projector"
    _auto_class = "AutoConfig"

    def __init__(
        self,
        visual_hidden_size=1024,
        llm_hidden_size=1024,
        depth=2,
        hidden_act="gelu",
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        super().__init__(**kwargs)


class VectorLLMConfig(PretrainedConfig):
    model_type = "vectorllm_hf_0407"
    processor_class = "VectorLLMProcessor"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        regression_size=(128, 128),
        projector_depth=2,
        visual_hidden_size=None,
        pixel_idx=0,
        pre_resize_size=432,
        resized_size=128,
        patch_size=16,
        do_normalize=False,
        vision_model_name_or_path="",
        llm_name_or_path="",
        visual_peft_config=None,
        vision_torch_dtype="bfloat16",
        **kwargs,
    ):
        serialized_visual_hidden_size = kwargs.get("vision_hidden_size", None)
        serialized_projector_config = kwargs.get("projector_config", None)
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
        if llm_config is None:
            llm_config = {}

        if isinstance(vision_config, HFRADIOConfig):
            vision_config = vision_config.to_dict()
        else:
            vision_config = copy.deepcopy(vision_config)
        if isinstance(llm_config, Qwen3Config):
            llm_config = llm_config.to_dict()
        else:
            llm_config = copy.deepcopy(llm_config)

        self.vision_config = vision_config
        self.llm_config = llm_config

        qwen3_config = Qwen3Config(**llm_config)
        radio_config = HFRADIOConfig(**vision_config)
        self.text_config = qwen3_config
        self.hidden_size = qwen3_config.hidden_size
        radio_args = radio_config.args or {}
        if visual_hidden_size is None and serialized_visual_hidden_size is not None:
            visual_hidden_size = serialized_visual_hidden_size
        self.vision_hidden_size = (
            visual_hidden_size
            if visual_hidden_size is not None
            else radio_args.get("mlp_hidden_size", qwen3_config.hidden_size)
        )

        if serialized_projector_config is not None:
            self.projector_config = copy.deepcopy(serialized_projector_config)
        else:
            self.projector_config = ProjectorConfig(
                visual_hidden_size=self.vision_hidden_size,
                llm_hidden_size=self.hidden_size,
                depth=projector_depth,
            ).to_dict()

        self.regression_size = tuple(regression_size)
        self.pixel_idx = pixel_idx
        self.tie_word_embeddings = False
        self.num_cls_register_tokens = 1 + radio_args.get("register_multiple", 0)
        self.pre_resize_size = pre_resize_size
        self.resized_size = resized_size
        self.patch_size = patch_size
        self.do_normalize = do_normalize
        self.vision_model_name_or_path = vision_model_name_or_path
        self.llm_name_or_path = llm_name_or_path
        self.visual_peft_config = copy.deepcopy(visual_peft_config)
        self.vision_torch_dtype = vision_torch_dtype

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = copy.deepcopy(self.vision_config)
        output["llm_config"] = copy.deepcopy(self.llm_config)
        output["text_config"] = self.text_config.to_dict()
        output["projector_config"] = copy.deepcopy(self.projector_config)
        output["model_type"] = self.__class__.model_type
        return output
