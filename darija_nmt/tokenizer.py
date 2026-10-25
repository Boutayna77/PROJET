"""SentencePiece wrapper (separate English / Arabic-script-Darija subword vocabularies)."""
from __future__ import annotations

from pathlib import Path

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class SPTokenizer:
    def __init__(self, model_path: str | Path):
        import sentencepiece as spm  # lazy import so text utilities work without it
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        assert (self.sp.pad_id(), self.sp.unk_id(), self.sp.bos_id(), self.sp.eos_id()) == (PAD, UNK, BOS, EOS)
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = PAD, UNK, BOS, EOS

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids if int(i) not in (PAD, UNK, BOS, EOS)])


def train_spm(input_file: str, prefix: str, vocab_size: int, model_type: str = "unigram",
              normalization: str = "identity") -> None:
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(
        input=input_file, model_prefix=prefix, vocab_size=vocab_size, model_type=model_type,
        character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS,
        normalization_rule_name=normalization, split_digits=True, hard_vocab_limit=False,
        minloglevel=2,
    )
