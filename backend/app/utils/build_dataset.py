import logging
from dataset_builder.dataset_builder import DatasetBuilder
from training.trainer import Trainer
from dotenv import load_dotenv, find_dotenv
from utils.utils import get_device

load_dotenv(find_dotenv())

if __name__ == "__main__":
    # Note: Multiprocessing is now disabled when using CUDA
    # GPU processing is sequential as GPU is already massively parallel

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    device = get_device()

    # builder = DatasetBuilder(mode='test', num_workers=8, device=device)
    # builder.build_complete_dataset(name="test_preprocessed_dataset")
    
    builder = DatasetBuilder(mode='train', num_workers=8, device=device)
    builder.build_complete_dataset(name="preprocessed_dataset")
    