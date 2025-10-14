import argparse

from src.search import searching_tool
from src.configurations import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search Environment"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        type=str,
        help="Config file path (default: None)"
    )

    args = parser.parse_args()

    # Validate required arguments
    if args.config is None:
        raise ValueError("The config argument should be set!")

    config = Config.from_yaml(args.config)

    results = searching_tool(query="سعدی در قرن چند میزیست؟", config=config)
    print(results)
