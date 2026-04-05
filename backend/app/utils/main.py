from logging import basicConfig, StreamHandler, FileHandler, getLogger, info
import torch
import argparse
from dataset_builder.dataset_builder import DatasetBuilder
from training.trainer import Trainer
from dotenv import load_dotenv, find_dotenv
from utils.utils import get_device
from core.model import DeepFakeDetector
from torchsummary import summary
import time
from datetime import datetime
from pathlib import Path
from inference import InferenceEngine
from utils import ConfigService

SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def retrieve_args() -> argparse.Namespace:
    """Retrieve command-line arguments.
    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='VeridisQuo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='path to configuration file (JSON or YAML)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'build_dataset'],
        default='inference',
        help='execution mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='input file or directory for inference'
    )
    return parser.parse_args()
    
def init_logging_system(
    log_dir: str,
    log_filename: str, 
    log_level: int,
    log_format: str,
    log_datefmt: str,
    console: bool =True,
    file: bool =True
):
    #create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    handlers = []
    if console:
        handlers.append(StreamHandler())
    if file:
        handlers.append(FileHandler(log_filename, mode='w', encoding='utf-8'))
        
    basicConfig(
        filename=log_filename, 
        level=log_level, 
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers
    )

def check_input_file(input_path: Path) -> bool:
    """Check if the input file is a video or an image based on its extension.
    Parameters:
        input_path (Path): path to the input file
    Returns:
        bool: True if video, False otherwise
    Raises:
        ValueError: if file extension is unsupported
    """
    assert input_path.is_file(), f"Input path is not a file: {input_path}"
    assert input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS.extend(SUPPORTED_IMAGE_EXTENSIONS), \
        "Unsupported file extension: {input_path.suffix}. Supported extensions: " + \
        f"Videos: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}; Images: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        
    return input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        

if __name__ == "__main__":
    #load environment variables
    load_dotenv(find_dotenv())

    #retrieve command-line arguments
    args = retrieve_args()
    try:
        assert args.config, "No configuration file provided. Use --config to specify the config file."
        assert Path(args.config).exists(), f"Configuration file not found: {args.config}"
    
        config: ConfigService = ConfigService(ConfigService.load_config(args.config))
    
        assert config.get('type') in args.mode or config.get('type') is None, "Configuration type does not match execution mode."
    
    except AssertionError as e:
        print(f"Could not start VeridisQuo, invalid parameter: {e}")
        exit(1)

    try: 
        log_dir = config.get('paths').get('output').get('logs_dir')
        log_filename = log_dir + '/' + config.get('logging').get('file').get('path')
        log_level = config.get('logging').get('level')
        log_format = config.get('logging').get('format')
        log_datefmt = config.get('logging').get('datefmt')
        log_console_enabled = config.get('logging').get('console').get('enabled')
        log_file_enabled = config.get('logging').get('file').get('enabled')
        
        init_logging_system(
            log_dir=log_dir,
            log_filename=log_filename,
            log_level=log_level,
            log_format=log_format,
            log_datefmt=log_datefmt,
            console=log_console_enabled,
            file=log_file_enabled
        )

        logger = getLogger("/".join(__file__.split("/")[-2:]))
        logger.info(f"Logging initialized to file: {log_filename}")
        
        #execute based on mode
        match args.mode:
            case 'inference':
                logger.info("Starting inference mode...")

                assert args.input, "Input path is required for inference mode."
                
                engine = InferenceEngine(config=config)

                input_path = Path(args.input)
                assert input_path.exists(), f"Input path not found: {input_path}"

                aggregation_method=config.get('inference').get('score_aggregation').get('aggregation_method')
                batch_size=config.get('inference').get('batch_processing').get('default_batch_size')
                frames_per_second=config.get('preprocessing').get('frame_extraction').get('frames_per_second')
                use_pyav_extractor=config.get('preprocessing').get('frame_extraction').get('use_optimized')               
                
                if input_path.is_file():
                    logger.info(f"Running inference on file: {input_path}")
                    check_input_file(input_path)
                    
                    result = engine.predict_video(
                        video_paths=input_path,
                        batch_size=batch_size,
                        frames_per_second=frames_per_second,
                        use_pyav_extractor=use_pyav_extractor,
                        aggregation_method=aggregation_method
                    )
                    
                elif input_path.is_dir():
                    input_files = list(input_path.glob('*'))
                    logger.info(f"Running inference on directory: {input_path} ({len(input_files)} files found)")
                    
                    for file in input_files:
                        check_input_file(file)
                    
                    result = engine.predict_video_batch(
                        video_paths=input_files,
                        batch_size=batch_size,
                        frames_per_second=frames_per_second,
                        use_pyav_extractor=use_pyav_extractor,
                        aggregation_method=aggregation_method
                    )

            case 'train':
                logger.info("Starting training mode...")

                #TODO: initialize trainer with config
                # trainer = Trainer(config=config)
                # trainer.train()

                logger.warning("Training mode not yet implemented")

            case 'build_dataset':
                logger.info("Starting dataset building mode...")

                #TODO: initialize dataset builder with config
                # builder = DatasetBuilder(config=config)
                # builder.build()

                logger.warning("Dataset building mode not yet implemented")

        logger.info("Execution completed successfully")
        
    except KeyError as e:
        err_mess = f"Configuration key error: {e}"
    
    except AssertionError as e:
        err_mess = f"Invalid data: {e}"
    
    except Exception as e:
        err_mess = f"An unexpected error occurred: {e}"
        
    finally:
        logger.fatal(err_mess)
        print(err_mess)
        exit(1)