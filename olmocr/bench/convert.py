import argparse
import os
import glob
import importlib
from tqdm import tqdm

def parse_method_arg(method_arg):
    """
    Parse a method configuration string of the form:
       method_name[:key=value[:key2=value2...]]
    Returns:
       (method_name, kwargs_dict)
    """
    parts = method_arg.split(":")
    name = parts[0]
    kwargs = {}
    for extra in parts[1:]:
        if "=" in extra:
            key, value = extra.split("=", 1)
            try:
                converted = int(value)
            except ValueError:
                try:
                    converted = float(value)
                except ValueError:
                    converted = value
            kwargs[key] = converted
        else:
            raise ValueError(f"Extra argument '{extra}' is not in key=value format")
    return name, kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PDF conversion using specified OCR methods and extra parameters."
    )
    parser.add_argument(
        "methods",
        nargs="+",
        help="Methods to run in the format method[:key=value ...]. "
             "Example: gotocr mineru:temperature=2 marker:runs=3"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat the conversion for each PDF."
    )
    args = parser.parse_args()

    # Mapping of method names to a tuple: (module path, function name)
    available_methods = {
        "gotocr": ("olmocr.bench.runners.run_gotocr", "run_gotocr"),
        "marker": ("olmocr.bench.runners.run_marker", "run_marker"),
        "mineru": ("olmocr.bench.runners.run_mineru", "run_mineru"),
        "chatgpt": ("olmocr.bench.runners.run_chatgpt", "run_chatgpt"),
    }

    # Build config by importing only requested methods.
    config = {}
    for method_arg in args.methods:
        method_name, extra_kwargs = parse_method_arg(method_arg)
        if method_name not in available_methods:
            parser.error(f"Unknown method: {method_name}. "
                         f"Available methods: {', '.join(available_methods.keys())}")
        module_path, function_name = available_methods[method_name]
        # Dynamically import the module and get the function.
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        config[method_name] = {
            "method": function,
            "kwargs": extra_kwargs
        }

    data_directory = os.path.join(os.path.dirname(__file__), "sample_data")
    pdf_directory = os.path.join(data_directory, "pdfs")

    # Process each PDF using each specified method and repeat the conversion as needed.
    for candidate in config.keys():
        print(f"Starting conversion using {candidate} with kwargs: {config[candidate]['kwargs']}")
        candidate_output_dir = os.path.join(data_directory, candidate)
        os.makedirs(candidate_output_dir, exist_ok=True)

        for pdf_path in tqdm(glob.glob(os.path.join(pdf_directory, "*.pdf")), desc=candidate):
            base_name = os.path.basename(pdf_path).replace(".pdf", "")
            # Repeat the conversion as many times as specified.
            for i in range(1, args.repeats + 1):
                markdown = config[candidate]["method"](pdf_path, page_num=1, **config[candidate]["kwargs"])
                output_filename = f"{base_name}_{i}.md"
                output_path = os.path.join(candidate_output_dir, output_filename)
                with open(output_path, "w") as out_f:
                    out_f.write(markdown)
