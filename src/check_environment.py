import sys

import pandas


def main() -> None:
    print(sys.executable)
    print(sys.version)
    print(pandas.__version__)


if __name__ == "__main__":
    main()
