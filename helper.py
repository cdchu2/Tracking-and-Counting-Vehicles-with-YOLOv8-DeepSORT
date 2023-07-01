import cv2
from typing import Any

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

def cornerRect(img: Any, bbox: Any, l: int = 30, t: int = 5, rt: int = 1,
                colorR: Any = (255, 255, 255), colorC: Any = (0, 0, 255)) -> Any:
    x, y, w, h = bbox

    # cv2.rectangle(img, (x, y), (x+w, y+h), colorR, rt)

    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    cv2.line(img, (x + w, y), (x + w - l, y), colorC, t)
    cv2.line(img, (x + w, y), (x + w, y + l), colorC, t)
    cv2.line(img, (x, y + h), (x + l, y + h), colorC, t)
    cv2.line(img, (x, y + h), (x, y + h - l), colorC, t)
    cv2.line(img, (x + w, y + h), (x + w - l, y + h), colorC, t)
    cv2.line(img, (x + w, y + h), (x + w, y + h - l), colorC, t)

    return img