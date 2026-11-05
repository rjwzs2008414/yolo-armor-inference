"""Simulate timeout and unsupported pixel format, and log cleanup steps."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from camera.mvs_camera import MvsCamera


def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"logs/error_{time.strftime('%Y%m%d_%H%M%S')}.log",
                                encoding="utf-8"),
        ],
    )


def test_timeout() -> None:
    cam = MvsCamera(0)
    try:
        cam.open()
        cam.set_pixel_format("BayerRG8")
        cam.set_exposure_us(500)
        cam.set_gain_db(0)
        cam.start()
        logging.info("timeout test: started grabbing")
        for i in range(3):
            try:
                cam.read_raw(timeout_ms=1)
                logging.info("timeout test: got frame %d (no timeout)", i)
            except TimeoutError as exc:
                logging.info("timeout test: expected timeout: %s", exc)
                break
    except Exception:
        logging.exception("timeout test failed")
    finally:
        logging.info("timeout test: stopping grabbing")
        cam.close()
        logging.info("timeout test: StopGrabbing/CloseDevice/DestroyHandle done")


def test_bad_format() -> None:
    cam = MvsCamera(0)
    try:
        cam.open()
        logging.info("format test: camera opened")
        try:
            cam.set_pixel_format("BGR8")
            logging.info("format test: BGR8 is supported")
        except Exception as exc:
            logging.info("format test: BGR8 unsupported: %s", exc)
        # Force an invalid enum value to exercise the error path.
        invalid = 0xFFFFFFFF
        ret = cam.cam.MV_CC_SetEnumValue("PixelFormat", invalid)
        logging.info("format test: invalid PixelFormat return=0x%08X", ret)
        if ret == 0:
            raise RuntimeError("camera unexpectedly accepted invalid PixelFormat")
    except Exception:
        logging.exception("format test: expected failure captured")
    finally:
        logging.info("format test: closing camera")
        cam.close()
        logging.info("format test: StopGrabbing/CloseDevice/DestroyHandle done")


if __name__ == "__main__":
    setup_logging()
    test_timeout()
    test_bad_format()
