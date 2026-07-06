"""CLI entry point for the detector evaluation harness (moved into the package).

The reusable IoU evaluation machinery (load_split / evaluate / make_detfn / print_report / match /
``compare``) now lives in ``umiusi_perception.eval`` so it is importable by the package's consumers
(training, the campaign eval scripts) without a tool-imports-tool hop. This file stays as the
``python -m tools.perception_eval`` command.

    python -m tools.perception_eval --method all      # color / hough / combined, per-colour P/R/F1
"""

import sys

from umiusi_perception.eval import main

if __name__ == "__main__":
    sys.exit(main())
