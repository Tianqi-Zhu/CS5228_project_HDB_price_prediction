"""Exclusive process lock for the archived runner on Windows and POSIX."""
import os

LOCK_EX = 2
LOCK_NB = 4


def flock(stream, operation):
    if operation not in (LOCK_EX, LOCK_EX | LOCK_NB):
        raise ValueError("Only exclusive acquisition is used by this runner")
    if os.name != "nt":
        import fcntl
        return fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if operation & LOCK_NB else 0))
    import msvcrt
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write("0")
        stream.flush()
    stream.seek(0)
    # Closing the stream releases the lock on both supported platforms.
    return msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK if operation & LOCK_NB else msvcrt.LK_LOCK, 1)
