"""Install checksum-pinned upstream replay fixes and experiment checks."""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def install(root, bundle):
    manifest = json.loads((bundle / "overlay-manifest.json").read_text())
    for name, entry in manifest["files"].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != entry["base_sha256"]:
            raise ValueError("Replay overlay base source changed: " + name)
        contents = (bundle / "overlay" / name).read_bytes()
        if hashlib.sha256(contents).hexdigest() != entry["installed_sha256"]:
            raise ValueError("Replay overlay checksum mismatch: " + name)
        path.write_bytes(contents)
    for source, target in [
        ("replay_runtime.py", "snowball_replay_runtime.py"),
        ("parity_runtime.py", "snowball_parity_runtime.py"),
    ]:
        shutil.copyfile(bundle / source, root / "skyrl-train" / target)
    (root / "snowball-replay-overlay.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Installed verified route-replay fixes and training checks", flush=True)


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
