"""Hikrobot MVS camera wrapper.

The wrapper keeps the MVS SDK details in one place:
  - enumerate/open/close device
  - exposure, gain and pixel format
  - grab one frame with MV_CC_GetImageBuffer
  - BayerRG8 -> BGR conversion with OpenCV

It is written for the Linux MVS SDK (tested path: /opt/MVS).
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import time
from ctypes import POINTER, byref, cast, memset, sizeof, string_at
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# MvCameraControl_class.py loads libMvCameraControl.so from
# $MVCAM_COMMON_RUNENV/64/libMvCameraControl.so.
_DEFAULT_RUNENV = "/opt/MVS/lib"
os.environ.setdefault("MVCAM_COMMON_RUNENV", _DEFAULT_RUNENV)

_MV_IMPORT = Path(__file__).resolve().parent.parent / "third_party" / "MvImport"
if str(_MV_IMPORT) not in sys.path:
    sys.path.insert(0, str(_MV_IMPORT))

from MvCameraControl_class import *  # noqa: E402,F401,F403
from CameraParams_header import MV_CC_DEVICE_INFO  # noqa: E402
from PixelType_header import (  # noqa: E402
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_RGB8_Packed,
)


def _decode_char(value) -> str:
    raw = bytes(value)
    raw = raw.split(b"\x00", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


@dataclass(frozen=True)
class CameraInfo:
    index: int
    model: str
    serial: str
    firmware: str
    interface: str
    user_name: str = ""


@dataclass
class Frame:
    bgr: np.ndarray
    raw: np.ndarray
    pixel_format: str
    width: int
    height: int
    frame_num: int
    host_time: float
    convert_ms: float


class MvsCamera:
    """One Hikrobot camera, 8N1-style simple synchronous grab API."""

    PIXEL_FORMATS = {
        "BayerRG8": PixelType_Gvsp_BayerRG8,
        "BGR8": PixelType_Gvsp_BGR8_Packed,
        "RGB8": PixelType_Gvsp_RGB8_Packed,
    }

    def __init__(self, index: int = 0):
        self.index = index
        self.cam = None
        self.info: Optional[CameraInfo] = None
        self.pixel_format = "BayerRG8"
        self.opened = False
        self.grabbing = False
        self._sdk_initialized = False

    @staticmethod
    def list_devices() -> list[CameraInfo]:
        MvCamera.MV_CC_Initialize()
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            layer_type = MV_USB_DEVICE | MV_GIGE_DEVICE
            ret = MvCamera.MV_CC_EnumDevices(layer_type, device_list)
            if ret != 0:
                raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08X}")
            devices: list[CameraInfo] = []
            for i in range(device_list.nDeviceNum):
                info = cast(
                    device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
                ).contents
                if info.nTLayerType == MV_USB_DEVICE:
                    usb = info.SpecialInfo.stUsb3VInfo
                    devices.append(
                        CameraInfo(
                            index=i,
                            model=_decode_char(usb.chModelName),
                            serial=_decode_char(usb.chSerialNumber),
                            firmware=_decode_char(usb.chDeviceVersion),
                            interface="USB3",
                            user_name=_decode_char(usb.chUserDefinedName),
                        )
                    )
                elif info.nTLayerType == MV_GIGE_DEVICE:
                    gige = info.SpecialInfo.stGigEInfo
                    devices.append(
                        CameraInfo(
                            index=i,
                            model=_decode_char(gige.chModelName),
                            serial=_decode_char(gige.chSerialNumber),
                            firmware=_decode_char(gige.chDeviceVersion),
                            interface="GigE",
                            user_name=_decode_char(gige.chUserDefinedName),
                        )
                    )
            return devices
        finally:
            MvCamera.MV_CC_Finalize()

    def open(self) -> None:
        if self.opened:
            return
        MvCamera.MV_CC_Initialize()
        self._sdk_initialized = True
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            MV_USB_DEVICE | MV_GIGE_DEVICE, device_list
        )
        if ret != 0:
            self.close()
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08X}")
        if device_list.nDeviceNum == 0:
            self.close()
            raise RuntimeError("no MVS camera found")
        if self.index >= device_list.nDeviceNum:
            self.close()
            raise RuntimeError(
                f"camera index {self.index} out of range "
                f"(found {device_list.nDeviceNum})"
            )

        dev_info = cast(
            device_list.pDeviceInfo[self.index], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            self.close()
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:08X}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.close()
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:08X}")
        self.opened = True

        # Turn off automatic exposure/gain so the requested values are real.
        for key, value in (("ExposureAuto", 0), ("GainAuto", 0)):
            try:
                self.cam.MV_CC_SetEnumValue(key, value)
            except Exception:
                pass

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.close()
            raise RuntimeError(f"set TriggerMode failed: 0x{ret:08X}")

        # GigE cameras need a packet size; USB cameras ignore this failure.
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if packet_size > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

        self.info = self._read_info(dev_info)

    @staticmethod
    def _read_info(dev_info) -> CameraInfo:
        if dev_info.nTLayerType == MV_USB_DEVICE:
            usb = dev_info.SpecialInfo.stUsb3VInfo
            return CameraInfo(
                index=0,
                model=_decode_char(usb.chModelName),
                serial=_decode_char(usb.chSerialNumber),
                firmware=_decode_char(usb.chDeviceVersion),
                interface="USB3",
                user_name=_decode_char(usb.chUserDefinedName),
            )
        gige = dev_info.SpecialInfo.stGigEInfo
        return CameraInfo(
            index=0,
            model=_decode_char(gige.chModelName),
            serial=_decode_char(gige.chSerialNumber),
            firmware=_decode_char(gige.chDeviceVersion),
            interface="GigE",
            user_name=_decode_char(gige.chUserDefinedName),
        )

    def set_pixel_format(self, name: str) -> None:
        if name not in self.PIXEL_FORMATS:
            raise ValueError(f"unsupported pixel format: {name}")
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", self.PIXEL_FORMATS[name])
        if ret != 0:
            raise RuntimeError(
                f"camera does not accept PixelFormat={name}: 0x{ret:08X}"
            )
        self.pixel_format = name

    def set_exposure_us(self, value: float) -> None:
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(value))
        if ret != 0:
            raise RuntimeError(f"set ExposureTime={value} failed: 0x{ret:08X}")

    def set_gain_db(self, value: float) -> None:
        ret = self.cam.MV_CC_SetFloatValue("Gain", float(value))
        if ret != 0:
            raise RuntimeError(f"set Gain={value} failed: 0x{ret:08X}")

    def start(self) -> None:
        if self.grabbing:
            return
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:08X}")
        self.grabbing = True

    def stop(self) -> None:
        if not self.grabbing:
            return
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StopGrabbing failed: 0x{ret:08X}")
        self.grabbing = False

    def read_raw(self, timeout_ms: int = 1000):
        if not self.grabbing:
            self.start()
        st_frame = MV_FRAME_OUT()
        memset(byref(st_frame), 0, sizeof(st_frame))
        ret = self.cam.MV_CC_GetImageBuffer(st_frame, int(timeout_ms))
        if ret != 0:
            raise TimeoutError(f"MV_CC_GetImageBuffer timeout/error: 0x{ret:08X}")
        try:
            info = st_frame.stFrameInfo
            width, height = int(info.nWidth), int(info.nHeight)
            frame_len = int(info.nFrameLen)
            if not st_frame.pBufAddr:
                raise RuntimeError("MVS returned an empty frame buffer")
            data = string_at(st_frame.pBufAddr, frame_len)
            raw = np.frombuffer(data, dtype=np.uint8).copy()
            if self.pixel_format == "BayerRG8":
                raw = raw.reshape(height, width)
            elif self.pixel_format in ("BGR8", "RGB8"):
                raw = raw.reshape(height, width, 3)
            else:
                raw = raw.reshape(height, width)
            return raw, info
        finally:
            self.cam.MV_CC_FreeImageBuffer(st_frame)

    def read(self, timeout_ms: int = 1000) -> Frame:
        raw, info = self.read_raw(timeout_ms)
        started = time.perf_counter()
        if self.pixel_format == "BayerRG8":
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
        elif self.pixel_format == "RGB8":
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = raw.copy()
        convert_ms = (time.perf_counter() - started) * 1000.0
        return Frame(
            bgr=bgr,
            raw=raw,
            pixel_format=self.pixel_format,
            width=int(info.nWidth),
            height=int(info.nHeight),
            frame_num=int(info.nFrameNum),
            host_time=time.time(),
            convert_ms=convert_ms,
        )

    def close(self) -> None:
        if self.cam is not None:
            if self.grabbing:
                try:
                    self.cam.MV_CC_StopGrabbing()
                except Exception:
                    pass
                self.grabbing = False
            if self.opened:
                try:
                    self.cam.MV_CC_CloseDevice()
                except Exception:
                    pass
                self.opened = False
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.cam = None
        if self._sdk_initialized:
            try:
                MvCamera.MV_CC_Finalize()
            except Exception:
                pass
            self._sdk_initialized = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def main() -> None:
    devices = MvsCamera.list_devices()
    print(f"found {len(devices)} device(s)")
    for d in devices:
        print(
            f"[{d.index}] {d.interface} {d.model} "
            f"serial={d.serial} firmware={d.firmware}"
        )
    if not devices:
        return
    with MvsCamera(devices[0].index) as cam:
        cam.set_pixel_format("BayerRG8")
        cam.set_exposure_us(1000)
        cam.set_gain_db(0)
        frame = cam.read()
        print(
            f"frame {frame.width}x{frame.height} "
            f"format={frame.pixel_format} "
            f"convert={frame.convert_ms:.3f}ms "
            f"mean={frame.bgr.mean():.1f}"
        )


if __name__ == "__main__":
    main()
