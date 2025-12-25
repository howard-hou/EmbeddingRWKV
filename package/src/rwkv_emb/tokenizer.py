import os
from .reference.utils import TRIE_TOKENIZER

class RWKVTokenizer(TRIE_TOKENIZER):
    def __init__(self, vocab_path=None):
        """
        初始化 Tokenizer，默认自动寻找包内的 vocab 文件。
        """
        if vocab_path is None:
            # 获取当前文件 (tokenizer.py) 的目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 推算 vocab 文件的相对路径
            vocab_path = os.path.join(current_dir, "reference", "rwkv_vocab_v20230424.txt")
            
        super().__init__(vocab_path)
        self.eos_token_id = 65535

    def encode(self, text: str, add_eos: bool = True):
        """
        增强的 encode 方法，支持自动添加 EOS
        """
        ids = super().encode(text)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        return super().decode(ids)