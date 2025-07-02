import os
import cv2
import glob
import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse

# custom YAML dumper to force inline lists for "data" fields
class InlineListDumper(yaml.SafeDumper):
    pass

def represent_inline_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

InlineListDumper.add_representer(list, represent_inline_list)

def save_ros_yaml(cameraMatrix, distCoeffs, image_size, save_path="configs/example.yaml"):
    fx = float(cameraMatrix[0, 0])
    fy = float(cameraMatrix[1, 1])
    cx = float(cameraMatrix[0, 2])
    cy = float(cameraMatrix[1, 2])

    ros_yaml = {
        "camera_name": "my_camera",
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                round(fx, 8), 0.0, round(cx, 8),
                0.0, round(fy, 8), round(cy, 8),
                0.0, 0.0, 1.0
            ]
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [round(float(x), 8) for x in distCoeffs.flatten()]
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                round(fx, 8), 0.0, round(cx, 8), 0.0,
                0.0, round(fy, 8), round(cy, 8), 0.0,
                0.0, 0.0, 1.0, 0.0
            ]
        }
    }

    with open(save_path, 'w') as f:
        yaml.dump(ros_yaml, f, Dumper=InlineListDumper, default_flow_style=False, sort_keys=False)

    print(f"Saved to {save_path} in ROS camera_info format.")

def show_image(title, image, figsize=(16, 12)):
    """
    Display an image in a larger matplotlib window.

    Args:
        title (str): Title of the image window.
        image (np.ndarray): Image in BGR format.
        figsize (tuple): Size of the figure in inches (width, height).
    """
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images(image_list, titles):
    n = len(image_list)
    cols = 6
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 2.5))

    for i, (img, title) in enumerate(zip(image_list, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=8)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.3,
                        left=0.02, right=0.98, top=0.95, bottom=0.05)
    plt.show()

def run(images, checkerboard, root):
    if not images:
        print("No images found.")
        exit(1)

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    imgs = []
    titles = []

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            for i, corner in enumerate(corners2):
                x, y = corner.ravel()
                cv2.putText(
                    img, str(i), (int(x), int(y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 255, 0),  # Green text
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
            cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
            show_image(idx, img) #debug
            imgs.append(img)
            titles.append(idx)
    show_images(imgs, titles)

    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera matrix (K):")
    print(cameraMatrix)
    print("\nDistortion coefficients (D):")
    print(distCoeffs.ravel())
    print(f"\nHeight : {img.shape[0]}, Width : {img.shape[1]}")

    for i in range(min(2, len(images))):
        img = cv2.imread(images[i])
        undistorted = cv2.undistort(img, cameraMatrix, distCoeffs, None)
        concat = np.hstack((img, undistorted))
        show_image(f"Distortion {i+1} (Original | Undistorted)", concat)

    save_ros_yaml(cameraMatrix=cameraMatrix,
                  distCoeffs=distCoeffs, image_size=(img.shape[0], img.shape[1]),
                  save_path=os.path.join('configs', f'{root}.yaml'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Intrinsic Calibration using Chessboard Images")
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--checkerboard', type=int, nargs=2, default=(6, 5),
                        help='Checkerboard dimensions (rows, cols)')
    parser.add_argument('--square_size', type=float, default=150.0,
                        help='Size of each square in mm')
    args = parser.parse_args()
    row, col = args.checkerboard[0], args.checkerboard[1]
    CHECKERBOARD = (col, row)
    SQUARE_SIZE = args.square_size / 1000.0  # Convert mm to meters

    if len(CHECKERBOARD) != 2:
        raise ValueError("Checkerboard must be specified as two integers (rows, cols).")
    if SQUARE_SIZE <= 0:
        raise ValueError("Square size must be a positive number.")
    print(f"Checkerboard: {CHECKERBOARD}, Square Size: {SQUARE_SIZE} m")

    # Find all images in the specified directory
    image_dir = os.path.join(args.root, 'Image')
    all_images = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) + glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    print(all_images)
    run(images=all_images, checkerboard=CHECKERBOARD, root=os.path.basename(args.root))
