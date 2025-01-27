import sys
import subprocess
import logging

logger = logging.getLogger(__name__)

def check_poppler_version():
    try:
        result = subprocess.run(['pdftoppm', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stderr.startswith("pdftoppm"):
            logger.info("pdftoppm is installed and working.")
        else:
            logger.error(f"pdftoppm is installed but returned an error.")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("pdftoppm is not installed.")
        logger.error("Check the README in the https://github.com/allenai/olmocr/blob/main/README.md for installation instructions")
        sys.exit(1)

if __name__ == "__main__":
    check_poppler_version()