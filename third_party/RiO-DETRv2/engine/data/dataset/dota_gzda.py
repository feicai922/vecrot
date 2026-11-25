"""gzda_v3 单类别（building）DOTA 数据集。

复用 DOTADataset 的解析逻辑，只把 CLASSES 改成单类，
以便直接训练 1 类建筑物旋转框检测。
"""
from .dota_dataset import DOTADataset
from ...core import register


@register()
class GzdaDOTADataset(DOTADataset):
    CLASSES = ('building',)
