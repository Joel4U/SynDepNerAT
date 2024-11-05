from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	pos: List[str]
	depheads: List[int]
	deplabels: List[str]
	span_labels: List[set] = None
