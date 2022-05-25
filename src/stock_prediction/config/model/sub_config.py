from dataclasses import dataclass


@dataclass(frozen=True)
class FetcherConfig:
    start_date: str
    end_date: str
    reg_exp: str
    normalization: bool


@dataclass(frozen=True)
class PreprocessorConfig:
    split_ratio: float
    tokenizer: str
    bpe_vocab_size: int
    min_vocab_frequency: int
    max_vocab_percentage: float


@dataclass(frozen=True)
class TrainerConfig:
    corpus_file_name: str
    dictionary_file_name: str


@dataclass(frozen=True)
class EvaluatorConfig:
    min_num_topics: int
    max_num_topics: int
    add_n_topics: int

# @dataclass(frozen=True)
# class PredictorConfig:
#     corpus_file_name: str
#     dictionary_file_name: str
