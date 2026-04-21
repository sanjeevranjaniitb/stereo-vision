import io
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def _img_from_b64(data: str) -> np.ndarray:
    raw = base64.b64decode(data.split(",")[1] if "," in data else data)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _to_b64_png(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _generate_stereo_pair(img: np.ndarray, shift: int = 30):
    """Simulate a stereo pair by horizontally shifting the image."""
    h, w = img.shape[:2]
    left = img[:, shift:, :]
    right = img[:, :w - shift, :]
    return left, right


def _compute_disparity(left: np.ndarray, right: np.ndarray,
                       num_disp: int = 128, block_size: int = 9):
    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo_l = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=32, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    stereo_r = cv2.ximgproc.createRightMatcher(stereo_l) if hasattr(cv2, "ximgproc") else None

    disp_l = stereo_l.compute(gl, gr).astype(np.float32) / 16.0

    if stereo_r is not None:
        disp_r = stereo_r.compute(gr, gl).astype(np.float32) / 16.0
        wls = cv2.ximgproc.createDisparityWLSFilter(stereo_l)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        disp_l = wls.filter(disp_l, gl, disparity_map_right=disp_r)

    return disp_l


def _colorize_disparity(disp: np.ndarray) -> np.ndarray:
    d = disp.copy()
    d[d < 0] = 0
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)


def _compute_depth_map(disp: np.ndarray, baseline: float = 0.12,
                       focal: float = 700.0) -> np.ndarray:
    safe = disp.copy()
    safe[safe <= 0] = 0.01
    return (baseline * focal) / safe


def _depth_to_pointcloud(depth: np.ndarray, color: np.ndarray,
                         focal: float = 700.0, step: int = 4):
    h, w = depth.shape[:2]
    points, colors = [], []
    cx, cy = w / 2, h / 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            z = depth[y, x]
            if z < 0.1 or z > 50:
                continue
            px = (x - cx) * z / focal
            py = (y - cy) * z / focal
            points.append([float(px), float(py), float(z)])
            b, g, r = color[y, x]
            colors.append([int(r), int(g), int(b)])
    return points, colors


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process():
    data = request.json
    img = _img_from_b64(data["image"])
    h, w = img.shape[:2]
    if w > 960:
        scale = 960 / w
        img = cv2.resize(img, (960, int(h * scale)))

    shift = int(data.get("shift", 30))
    num_disp = int(data.get("numDisparities", 128))
    block_size = int(data.get("blockSize", 9))
    if block_size % 2 == 0:
        block_size += 1

    left, right = _generate_stereo_pair(img, shift)
    disp = _compute_disparity(left, right, num_disp, block_size)
    color_disp = _colorize_disparity(disp)
    depth = _compute_depth_map(disp, baseline=shift / 250.0)
    points, colors = _depth_to_pointcloud(depth, left, step=3)

    return jsonify({
        "left": _to_b64_png(left),
        "right": _to_b64_png(right),
        "disparity": _to_b64_png(color_disp),
        "pointcloud": {"points": points, "colors": colors},
    })


@app.route("/api/triangulate", methods=["POST"])
def triangulate():
    d = request.json
    baseline = float(d.get("baseline", 0.12))
    focal = float(d.get("focal", 700))
    disparity = float(d.get("disparity", 30))
    if disparity == 0:
        disparity = 0.01
    z = (baseline * focal) / disparity
    return jsonify({"depth": round(z, 4), "baseline": baseline,
                    "focal": focal, "disparity": disparity})


if __name__ == "__main__":
    app.run(debug=True, port=5555)
