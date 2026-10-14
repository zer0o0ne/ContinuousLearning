"""Temporary, indexed disk storage for pipeline data (never model weights).

SQLite pages stay reclaimable by the OS; its userspace cache is capped at 8 MiB.
Only requested objects are deserialised. Pickles are written/read by this run,
not an interchange format. Spawned workers reopen the same file, never receive
the corpus through multiprocessing's pickle stream.
"""

import os
import pickle
import fcntl
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from collections.abc import Sequence, MutableMapping


@contextmanager
def _scratch_lock(parent):
    # Serialize discovery/creation, including the instant between creating a
    # directory and locking its owner file. The tiny lock file is metadata.
    with open(os.path.join(parent, ".working-data.lock"), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield

class DiskStore:
    def __init__(self, parent):
        os.makedirs(parent, exist_ok=True)
        with _scratch_lock(parent):
            self._create(parent)
        self.path = os.path.join(self._temporary.name, "objects.sqlite")
        self._conn = None
        try:
            self.connection.execute("CREATE TABLE objects (space TEXT, key BLOB, value BLOB, "
                                    "PRIMARY KEY (space, key)) WITHOUT ROWID")
        except BaseException:
            self.close()
            raise

    def _create(self, parent):
        # SIGKILL/OOM cannot run finally. Reap only our own unlocked scratch
        # directories; a concurrent live phase holds its lock throughout.
        for entry in os.scandir(parent):
            if not entry.name.startswith(".working-data-") or not entry.is_dir():
                continue
            try:
                with open(os.path.join(entry.path, "owner.lock"), "r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    shutil.rmtree(entry.path)
            except (BlockingIOError, FileNotFoundError):
                pass
        self._temporary = tempfile.TemporaryDirectory(prefix=".working-data-", dir=parent)
        self._lock = open(os.path.join(self._temporary.name, "owner.lock"), "w+")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @property
    def connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.path, timeout=120, isolation_level=None)
            self._conn.execute("PRAGMA cache_size=-8192")
            self._conn.execute("PRAGMA mmap_size=0")
            # Scratch data is replayable; durable label shards/checkpoints are
            # committed separately. No multi-GB WAL retained by long readers.
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=OFF")
        return self._conn

    def __getstate__(self):
        return {"path": self.path, "_conn": None, "_temporary": None, "_lock": None}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def sequence(self, name):
        return DiskSequence(self, name)

    def mapping(self, name):
        return DiskMapping(self, name)


class DiskMapping(MutableMapping):
    def __init__(self, store, name):
        self.store, self.name = store, name

    def __getitem__(self, key):
        row = self.store.connection.execute(
            "SELECT value FROM objects WHERE space=? AND key=?",
            (self.name, pickle.dumps(key, protocol=5))).fetchone()
        if row is None:
            raise KeyError(key)
        return pickle.loads(row[0])

    def __setitem__(self, key, value):
        self.store.connection.execute(
            "INSERT OR REPLACE INTO objects VALUES (?, ?, ?)",
            (self.name, pickle.dumps(key, protocol=5), pickle.dumps(value, protocol=5)))

    def __delitem__(self, key):
        cur = self.store.connection.execute(
            "DELETE FROM objects WHERE space=? AND key=?",
            (self.name, pickle.dumps(key, protocol=5)))
        if cur.rowcount == 0:
            raise KeyError(key)

    def __iter__(self):
        # Only keys, never payloads. The cursor must not span a write by a
        # worker, so materialise the small key index and close it first.
        rows = self.store.connection.execute(
            "SELECT key FROM objects WHERE space=?", (self.name,)).fetchall()
        return iter(pickle.loads(row[0]) for row in rows)

    def __len__(self):
        return self.store.connection.execute(
            "SELECT COUNT(*) FROM objects WHERE space=?", (self.name,)).fetchone()[0]

    def clear(self):
        self.store.connection.execute("DELETE FROM objects WHERE space=?", (self.name,))


class DiskSequence(Sequence):
    def __init__(self, store, name):
        self.data = DiskMapping(store, name)
        self._length = 0
        self._cached = None

    def __getstate__(self):
        return {"data": self.data, "_length": self._length, "_cached": None}

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._cached is None or self._cached[0] != index:
            self._cached = (index, self.data[index])
        return self._cached[1]

    def __iter__(self):
        try:
            for i in range(len(self)):
                yield self[i]
        finally:
            self._cached = None

    def __setitem__(self, index, value):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.data[index] = value
        if self._cached is not None and self._cached[0] == index:
            self._cached = None

    def append(self, value):
        self.data[self._length] = value
        self._length += 1

    def extend(self, values):
        for value in values:
            self.append(value)

    def clear(self):
        # Detach this handle. A spawned worker may still own the old length;
        # the shared file is deleted after workers join, never from the parent
        # while they are reading it.
        self._length = 0
        self._cached = None

    def release(self):
        self.data.clear()
        self.clear()


class DatasetView(Sequence):
    """Small index/field view; slicing reads only the requested batch."""
    def __init__(self, data, indices=None, field=None):
        self.data, self.indices, self.field = data, indices, field

    def __len__(self):
        return len(self.data) if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = self.data[index if self.indices is None else int(self.indices[index])]
        return item if self.field is None else item[self.field]

    def column(self, field):
        return DatasetView(self.data, self.indices, field)


def disk_labels(store, paths, load):
    """Import existing portable NPZ shards one at a time, preserving row order."""
    from utils import progress
    data = store.sequence("labels")
    for path in progress(paths, desc="index labels", unit="shard"):
        shard = load(path)
        data.extend(shard)
        del shard
    return data
