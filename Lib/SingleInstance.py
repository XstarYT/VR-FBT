"""Windows named-mutex guard preventing conflicting VR-FBT instances."""

from __future__ import annotations

import ctypes


ERROR_ALREADY_EXISTS = 183
DEFAULT_MUTEX_NAME = "Local\\VR-FBT-VRChat-Control-Center"


class SingleInstance:
    def __init__(self, name: str = DEFAULT_MUTEX_NAME):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_bool
        ctypes.set_last_error(0)
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), "Could not create the VR-FBT instance mutex")
        self._kernel32 = kernel32
        self._handle = handle
        self.acquired = ctypes.get_last_error() != ERROR_ALREADY_EXISTS
        if not self.acquired:
            self.close()

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
