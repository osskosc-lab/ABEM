from __future__ import annotations

import json

from abem.r2v_protocol import validate_r2v_protocol


if __name__ == "__main__":
    print(json.dumps(validate_r2v_protocol(), indent=2))
