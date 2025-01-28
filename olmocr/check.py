import sys
import subprocess
import logging
import importlib.util

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

def check_sglang_version():
    if importlib.util.find_spec("sglang") is None:
        logger.error(f"Please make sure sglang is installed according to the latest instructions here: https://docs.sglang.ai/start/install.html")
        logger.error("Sglang needs to be installed with a separate command in order to find all dependencies properly.")
        sys.exit(1)


if __name__ == "__main__":
    check_poppler_version()
    check_sglang_version()