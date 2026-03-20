#!/bin/bash
# Convert FoundationPose ONNX models to TensorRT engines.
# Place scorer_hwc.onnx and refiner_hwc.onnx in foundationpose_cpp/models/ then run:
#   bash tools/cvt_onnx2trt_models.bash          # default: fp16
#   bash tools/cvt_onnx2trt_models.bash fp16
#   bash tools/cvt_onnx2trt_models.bash fp32

set -e

PRECISION="${1:-fp16}"
if [[ "$PRECISION" != "fp16" && "$PRECISION" != "fp32" ]]; then
  echo "Usage: $0 [fp16|fp32]  (default: fp16)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

# trtexec: prefer PATH, then Docker/apt-built location
TRTEXEC=""
if command -v trtexec &>/dev/null; then
  TRTEXEC="trtexec"
elif [ -x "/usr/src/tensorrt/bin/trtexec" ]; then
  TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
  echo "Error: trtexec not found. Install TensorRT or run from Docker with TensorRT." >&2
  echo "" >&2
  echo "If TensorRT was installed via apt (libnvinfer-samples), build trtexec once:" >&2
  echo "  cd /usr/src/tensorrt/samples/trtexec" >&2
  echo "  sudo TRT_LIB_DIR=/usr/lib/x86_64-linux-gnu make" >&2
  echo "  (binary will be at /usr/src/tensorrt/bin/trtexec)" >&2
  exit 1
fi

PRECISION_FLAG=""
if [ "$PRECISION" = "fp16" ]; then
  PRECISION_FLAG="--fp16"
fi

SCORER_ONNX="${MODELS_DIR}/scorer_hwc.onnx"
REFINER_ONNX="${MODELS_DIR}/refiner_hwc.onnx"
SCORER_ENGINE="${MODELS_DIR}/scorer_hwc_dynamic_${PRECISION}.engine"
REFINER_ENGINE="${MODELS_DIR}/refiner_hwc_dynamic_${PRECISION}.engine"

for f in "$SCORER_ONNX" "$REFINER_ONNX"; do
  if [ ! -f "$f" ]; then
    echo "Error: missing $f (download ONNX from Google Drive and put in ${MODELS_DIR})" >&2
    exit 1
  fi
done

echo "Precision:     $PRECISION"
echo "Using trtexec: $TRTEXEC"
echo "Models dir:    $MODELS_DIR"
echo ""

echo "[1/2] Converting scorer_hwc.onnx -> $(basename "$SCORER_ENGINE") ..."
"$TRTEXEC" --onnx="$SCORER_ONNX" \
  --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
  --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  $PRECISION_FLAG \
  --saveEngine="$SCORER_ENGINE"

echo ""
echo "[2/2] Converting refiner_hwc.onnx -> $(basename "$REFINER_ENGINE") ..."
"$TRTEXEC" --onnx="$REFINER_ONNX" \
  --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
  --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
  $PRECISION_FLAG \
  --saveEngine="$REFINER_ENGINE"

echo ""
echo "Done. Engines: $SCORER_ENGINE, $REFINER_ENGINE"
