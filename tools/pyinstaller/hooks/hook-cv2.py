"""Collect OpenCV without its FFmpeg video I/O library.

The app only reads still images through OpenCV, so the FFmpeg backend
(opencv_videoio_ffmpeg, about 30 MB on Windows) is never loaded.
"""

from _trim import keep, upstream_hook

_upstream = upstream_hook("cv2")
hiddenimports = _upstream.get("hiddenimports", [])
datas = _upstream.get("datas", [])
binaries = keep(
    _upstream.get("binaries", []),
    lambda source, _dest: not source.name.startswith("opencv_videoio_ffmpeg"),
)
