from transformers import BertConfig


class NADIMARBERTConfig(BertConfig):
    model_type = "nadi_marbert"

    def __init__(
        self,
        num_labels: int = 21,
        bottleneck_dim: int = 384,
        classif_dropout_rate: float = 0.1,
        vatt_use_common_transform: bool = False,
        vatt_final_adapter: bool = False,
        vatt_positional_keys: str = 'sinosoid',
        vatt_debug_shapes: bool = False,
        vatt_bottleneck_dim: int = 384,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.bottleneck_dim = bottleneck_dim
        self.classif_dropout_rate = classif_dropout_rate
        self.vatt_use_common_transform = vatt_use_common_transform
        self.vatt_final_adapter = vatt_final_adapter
        self.vatt_positional_keys = vatt_positional_keys
        self.vatt_debug_shapes = vatt_debug_shapes
        self.vatt_bottleneck_dim = vatt_bottleneck_dim
