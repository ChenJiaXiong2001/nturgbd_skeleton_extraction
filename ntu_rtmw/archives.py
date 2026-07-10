import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from .constants import ARCHIVE_EXTENSIONS, NTU_RGB_ARCHIVE_NAMES


def ensure_dirs(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def list_archives(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and archive_suffix(path) in ARCHIVE_EXTENSIONS
    )


def expected_ntu_rgb_archives(root):
    root = Path(root)
    return [root / name for name in NTU_RGB_ARCHIVE_NAMES]


def check_ntu_rgb_archives(root, require_all=False):
    expected = expected_ntu_rgb_archives(root)
    present = [path for path in expected if path.exists()]
    missing = [path.name for path in expected if not path.exists()]
    print("NTU RGB archives: {}/32 present in {}".format(len(present), Path(root).resolve()), flush=True)
    if missing:
        print("Missing: {}".format(", ".join(missing)), flush=True)
        if require_all:
            raise SystemExit("Put all nturgbd_rgb_s001.zip ... nturgbd_rgb_s032.zip files in {}".format(root))
    return present, missing


def archive_suffix(path):
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    if name.endswith(".tar.bz2"):
        return ".tar.bz2"
    if name.endswith(".tar.xz"):
        return ".tar.xz"
    return path.suffix.lower()


def extract_all(archives_dir, extract_dir, skip_existing=True, check_expected=True):
    expected, _ = check_ntu_rgb_archives(archives_dir, require_all=False) if check_expected else ([], [])
    archives = expected or list_archives(archives_dir)
    if not archives:
        print("No archives found in {}".format(Path(archives_dir).resolve()), flush=True)
        return []

    extracted = []
    for archive in archives:
        target = Path(extract_dir) / extraction_dir_name(archive)
        marker = target / ".extracted"
        if skip_existing and marker.exists():
            print("skip extracted {}".format(archive.name), flush=True)
            extracted.append(target)
            continue
        target.mkdir(parents=True, exist_ok=True)
        print("extract {} -> {}".format(archive.name, target), flush=True)
        extract_one(archive, target)
        marker.write_text(str(archive), encoding="utf-8")
        extracted.append(target)
    return extracted


def remove_extracted(target, extract_dir):
    target = Path(target)
    if not target.exists():
        return
    root = Path(extract_dir).resolve()
    resolved = target.resolve()
    if resolved == root:
        raise SystemExit("Refusing to delete extraction root: {}".format(root))
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SystemExit("Refusing to delete outside extraction root: {}".format(resolved)) from exc
    print("delete extracted {}".format(resolved), flush=True)
    shutil.rmtree(resolved)


def extraction_dir_name(archive):
    name = Path(archive).name.lower()
    if name.startswith("nturgbd_rgb_s") and name.endswith(".zip"):
        return name.removeprefix("nturgbd_rgb_").removesuffix(".zip")
    return Path(archive).stem


def extract_one(archive, target):
    archive = Path(archive)
    suffix = archive_suffix(archive)
    if suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        return
    if suffix in {".tar", ".gz", ".tgz", ".bz2", ".xz", ".tar.gz", ".tar.bz2", ".tar.xz"}:
        with tarfile.open(archive) as tf:
            tf.extractall(target)
        return
    if suffix in {".rar", ".7z"}:
        seven_zip = shutil.which("7z") or shutil.which("7za")
        if not seven_zip:
            raise SystemExit("Install 7-Zip and add 7z.exe to PATH to extract {}".format(archive))
        subprocess.run([seven_zip, "x", str(archive), "-o{}".format(target), "-y"], check=True)
        return
    raise SystemExit("Unsupported archive: {}".format(archive))
