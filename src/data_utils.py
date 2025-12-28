import os
import shutil
import zipfile
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPrepTool:
    """
    A utility class for preparing training and validation datasets.
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize the DataPrepTool with a base directory.
        
        Args:
            base_dir: The root directory of the project (default: current directory).
        """
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"

    def clear_data_directories(self):
        """
        Clear all contents in data/train and data/val.
        Creates the directories if they do not exist.
        """
        for target in [self.train_dir, self.val_dir]:
            if target.exists():
                logger.info(f"Clearing directory: {target}")
                # We use shutil.rmtree and then recreate to ensure it's completely clean
                # but to be safer/finer, we can remove children
                for item in target.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete {item}: {e}")
            else:
                logger.info(f"Creating directory: {target}")
                target.mkdir(parents=True, exist_ok=True)
        logger.info("Data directories cleared successfully.")

    def unzip_to_train(self, zip_filename: str = "images.zip"):
        """
        Unzip a specified zip file into the data/train directory.
        
        Args:
            zip_filename: The name or path of the zip file to extract.
        """
        zip_path = Path(zip_filename)
        
        # If the path is relative, check it relative to base_dir
        if not zip_path.is_absolute():
            zip_path = self.base_dir / zip_filename

        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            return False

        logger.info(f"Extracting {zip_path} to {self.train_dir}...")
        try:
            self.train_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.train_dir)
            logger.info(f"Extraction completed: {len(list(self.train_dir.glob('*')))} items extracted.")
            return True
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return False

    def run_workflow(self, zip_filename: str = "images.zip"):
        """
        Combined workflow: 1. Clear directories, 2. Unzip images.
        """
        self.clear_data_directories()
        if self.unzip_to_train(zip_filename):
            logger.info("Workflow completed successfully.")
        else:
            logger.warning("Workflow completed with errors during extraction.")

if __name__ == "__main__":
    # Example usage:
    # python src/data_utils.py --clear --unzip images.zip
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset Preparation Tool")
    parser.add_argument("--clear", action="store_true", help="Clear data/train and data/val")
    parser.add_argument("--unzip", type=str, help="Path to images.zip to extract to data/train")
    parser.add_argument("--all", action="store_true", help="Run full workflow (clear + unzip images.zip)")
    
    args = parser.parse_args()
    tool = DataPrepTool()
    
    if args.all:
        tool.run_workflow(args.unzip if args.unzip else "images.zip")
    else:
        if args.clear:
            tool.clear_data_directories()
        if args.unzip:
            tool.unzip_to_train(args.unzip)
