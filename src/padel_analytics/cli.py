from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze padel footage and export prototype shot predictions."
    )
    parser.add_argument("video", type=Path, help="Path to the input padel match video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/shot_predictions.json"),
        help="Output JSON or CSV path.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=15,
        help="Frame sampling interval for the starter prototype.",
    )
    return parser.parse_args()


def analyze_video(video_path: Path, sample_every: int) -> list[dict[str, object]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    predictions: list[dict[str, object]] = []

    with tqdm(total=frame_count, desc="Scanning frames", unit="frame") as progress:
        frame_index = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break

            if frame_index % sample_every == 0:
                predictions.append(
                    {
                        "frame": frame_index,
                        "timestamp_sec": round(frame_index / fps, 3),
                        "shot_type": "unknown",
                        "player": None,
                        "confidence": 0.0,
                    }
                )

            frame_index += 1
            progress.update(1)

    capture.release()
    return predictions


def write_predictions(predictions: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        pd.DataFrame(predictions).to_csv(output_path, index=False)
        return

    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = analyze_video(args.video, args.sample_every)
    write_predictions(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

