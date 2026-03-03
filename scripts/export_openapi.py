import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.main import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FastAPI OpenAPI schema to disk.")
    parser.add_argument(
        "--output",
        default="docs/openapi.v1.json",
        help="Output path for OpenAPI JSON artifact.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    schema = app.openapi()
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote OpenAPI schema to {output_path}")


if __name__ == "__main__":
    main()
