"""Capture identical source evidence for tuned and warptiled experiments."""

import hashlib
import zipfile


def snapshot_sources(root, source_dir, archive_path):
    """Archive and build from the same bytes; return relative-path SHA256 values.

    Capture whole source directories so generated-header templates, search-space
    JSON, helper modules and third-party notices travel with each experiment.
    Build products and previous results are deliberately outside this list.
    """
    paths = [root / "CMakeLists.txt"]
    for folder in ("src", "benchmarks", "scripts", "tests", "third_party"):
        paths.extend(path for path in (root / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    captured = {path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(paths)}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in captured.items():
            archive.writestr(name, data)
            target = source_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return hashes
