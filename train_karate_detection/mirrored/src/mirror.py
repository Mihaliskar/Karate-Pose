from pathlib import Path
import cv2
import argparse


def mirror_images(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images.")

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        # Horizontal mirror
        mirrored = cv2.flip(image, 1)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), mirrored)

        print(f"{image_path.name} -> {output_path}")

    print("Finished mirroring images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Horizontally mirror all images in a folder."
    )

    parser.add_argument(
        "--input",
        default="dataset/input",
        help="Folder containing the original images"
    )

    parser.add_argument(
        "--output",
        default="dataset/mirrored",
        help="Folder where mirrored images will be saved"
    )

    args = parser.parse_args()

    mirror_images(args.input, args.output)