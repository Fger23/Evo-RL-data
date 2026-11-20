"""Lossless RGB image transport for RTC observations."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import av
import cv2
import numpy as np

from .protocol import OBSERVATION_IMAGE_CODEC_HEADER, OBSERVATION_PNG_HEADER


@dataclass(frozen=True)
class LosslessJPEGImage:
    data: bytes
    height: int
    width: int


@dataclass(frozen=True)
class LosslessPNGImage:
    data: bytes
    height: int
    width: int


def lossless_jpeg_available(*, encoder: bool) -> bool:
    name, mode = ("ljpeg", "w") if encoder else ("mjpeg", "r")
    try:
        av.Codec(name, mode)
    except ValueError:
        return False
    return True


def lossless_png_available() -> bool:
    return cv2.haveImageWriter(".png")


def check_observation_image_codec(codec: str, ready: dict[str, str | bytes]) -> None:
    if codec == "raw":
        return
    if codec == "lossless_png":
        if ready.get(OBSERVATION_PNG_HEADER) != "1":
            raise RuntimeError("The RTC server must support lossless PNG observation transport")
        if not lossless_png_available():
            raise RuntimeError("The client's OpenCV build is missing PNG support")
        return
    if ready.get(OBSERVATION_IMAGE_CODEC_HEADER) != codec:
        raise RuntimeError("The RTC server must support lossless JPEG observation transport")
    if not lossless_jpeg_available(encoder=True):
        raise RuntimeError("The client's PyAV build is missing the lossless JPEG encoder (ljpeg)")


def _encode_jpeg(rgb: np.ndarray) -> LosslessJPEGImage:
    height, width, _ = rgb.shape
    encoder = av.CodecContext.create("ljpeg", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "bgr24"
    encoder.time_base = Fraction(1, 1)
    encoder.thread_count = 1
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    # BGR is a channel permutation of RGB, so this conversion preserves every sample.
    (packet,) = encoder.encode(frame.reformat(format="bgr24"))
    return LosslessJPEGImage(bytes(packet), height, width)


def _encode_png(rgb: np.ndarray) -> LosslessPNGImage:
    height, width, _ = rgb.shape
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, packet = cv2.imencode(".png", bgr, (cv2.IMWRITE_PNG_COMPRESSION, 1))
    if not ok:
        raise ValueError("Could not encode lossless PNG observation image")
    return LosslessPNGImage(packet.tobytes(), height, width)


def encode_observation_images(observation: dict[str, Any], codec: str) -> dict[str, Any]:
    if codec == "raw":
        return observation
    encode_image = _encode_png if codec == "lossless_png" else _encode_jpeg
    encoded = observation.copy()
    for key, value in observation.items():
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.uint8
            and value.ndim == 3
            and value.shape[2] == 3
            and value.size
        ):
            image = encode_image(value)
            if len(image.data) < value.nbytes:
                encoded[key] = image
    return encoded


def _decode_jpeg(image: LosslessJPEGImage) -> np.ndarray:
    decoder = av.CodecContext.create("mjpeg", "r")
    decoder.thread_count = 1
    try:
        (frame,) = decoder.decode(av.Packet(image.data))
        if (frame.height, frame.width) != (image.height, image.width) or not frame.format.is_rgb:
            raise ValueError("Lossless JPEG dimensions or color format do not match the observation")
        return frame.to_ndarray(format="rgb24")
    except av.error.FFmpegError as exc:
        raise ValueError("Invalid lossless JPEG observation image") from exc


def _decode_png(image: LosslessPNGImage) -> np.ndarray:
    try:
        packet = np.frombuffer(image.data, dtype=np.uint8)
        bgr = cv2.imdecode(packet, cv2.IMREAD_UNCHANGED)
        if bgr is None or bgr.shape != (image.height, image.width, 3) or bgr.dtype != np.uint8:
            raise ValueError("Lossless PNG dimensions or color format do not match the observation")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except cv2.error as exc:
        raise ValueError("Invalid lossless PNG observation image") from exc


def decode_observation_images(observation: dict[str, Any]) -> dict[str, Any]:
    decoded = observation.copy()
    for key, value in observation.items():
        if isinstance(value, LosslessPNGImage):
            decoded[key] = _decode_png(value)
        elif isinstance(value, LosslessJPEGImage):
            decoded[key] = _decode_jpeg(value)
    return decoded
