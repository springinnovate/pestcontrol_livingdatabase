"""Freshness checks for the full PCLD export."""

import hashlib
from pathlib import Path
import zipfile


# Increment when the contents or format of the full export change.
EXPORT_FORMAT_VERSION = 1


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def download_revision(database_path):
    """Fingerprint the SQLite file and any uncheckpointed WAL using file metadata."""
    database_path = Path(database_path)
    database_identity = _file_identity(database_path.stat())
    wal_path = database_path.with_name(database_path.name + "-wal")
    try:
        wal_stat = wal_path.stat()
    except FileNotFoundError:
        wal_identity = None
    else:
        wal_identity = _file_identity(wal_stat) if wal_stat.st_size else None

    identity = (EXPORT_FORMAT_VERSION, database_identity, wal_identity)
    return hashlib.sha256(repr(identity).encode("ascii")).hexdigest()


def cached_zip_matches(path, revision):
    """Read the ZIP's revision comment without decompressing any CSV files."""
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.comment == revision.encode("ascii")
    except (FileNotFoundError, zipfile.BadZipFile):
        return False
