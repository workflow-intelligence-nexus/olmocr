from transformers import (
    AutoProcessor
)

from pdelfin.train.core.cli import make_cli
from pdelfin.train.core.config import TrainConfig

from .utils import (
    make_dataset
)



def main():
    train_config = make_cli(TrainConfig)  # pyright: ignore
    
    processor = AutoProcessor.from_pretrained(train_config.model.name_or_path)
    train_dataset, valid_dataset = make_dataset(train_config, processor)    

    print("Training dataset........")
    print(train_dataset)
    print("\n\n")

    print("Validation dataset........")
    print(valid_dataset)
    print("\n\n")

    print("Datasets loaded into hugging face cache directory")


if __name__ == "__main__":
    main()
