#!/bin/bash
###############################################################################
#  文件名: vevo_run.slurm
#  用途  : Spartan 上一次跑完 VEVO 的 timbre/style/voice/tts 四种推理
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -J vevo-sanity
#SBATCH -o /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim2341/haoguangzhou/logs/%x-%j.err
#SBATCH -A punim2341

set -euo pipefail

#############################
# ===== 路径与环境变量  =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
CODE_DIR="${PROJECT_ROOT}/voice/Amphion-VevoDev"
RUNS_ROOT="${PROJECT_ROOT}/runs"
CONDA_ENV=vevo_env

# 统一创建项目根 logs（给 Slurm -o/-e 用）
mkdir -p "${PROJECT_ROOT}/logs"

# 模型/权重缓存放项目目录（仅设置 HF_HOME，避免 transformers 弃用警告）
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

export TMPDIR=${TMPDIR:-/tmp}

# 线程亲和性与并发控制，降低 ORT/BLAS 亲和性报错与资源争用
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export OMP_PROC_BIND=${OMP_PROC_BIND:-false}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_NUM_THREADS=${ORT_NUM_THREADS:-1}

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_vevo_infer"
LOG_DIR="${RUN_DIR}/logs"       # 运行期快照日志（与 Slurm 日志不同）
OUT_DIR="${RUN_DIR}/outputs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${STAT_DIR}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# 确保 Python 能导入项目内模块（更稳健）
# 避免 set -u 下 PYTHONPATH 未定义导致报错
: "${PYTHONPATH:=}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

# 选用当前环境的 python 可执行文件，避免落到系统 python
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}" | tee -a "${LOG_DIR}/run-info.txt"
PYTHON_BIN="${CONDA_PREFIX:-}/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(which python)"
echo "PYTHON_BIN: $PYTHON_BIN" | tee -a "${LOG_DIR}/run-info.txt"

# （已移除）自动加载系统模块以避免噪声日志；如需可手动开启
# module -t avail ffmpeg   >/dev/null 2>&1 && module load ffmpeg   >/dev/null 2>&1 || true
# module -t avail espeak   >/dev/null 2>&1 && module load espeak   >/dev/null 2>&1 || true
# module -t avail espeak-ng>/dev/null 2>&1 && module load espeak-ng>/dev/null 2>&1 || true

#############################
# ===== 依赖版本预检  =====
#############################
echo "==== DEP CHECK ====" | tee -a "${LOG_DIR}/run-info.txt"
CURR_TRF=$("$PYTHON_BIN" - <<'PY'
try:
    import transformers as t
    print(getattr(t, '__version__', '0'))
except Exception:
    print('0')
PY
)
echo "transformers version: ${CURR_TRF}" | tee -a "${LOG_DIR}/run-info.txt"
case "$CURR_TRF" in
  4.41.*) echo "[dep] transformers is compatible" | tee -a "${LOG_DIR}/run-info.txt" ;;
  *)
    echo "[dep] fixing transformers to 4.41.2 (and tokenizers<0.20)" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install --upgrade "transformers==4.41.2" "tokenizers<0.20" | tee -a "${LOG_DIR}/run-info.txt"
    NEW_TRF=$("$PYTHON_BIN" -c "import transformers,sys; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    echo "[dep] transformers now: ${NEW_TRF}" | tee -a "${LOG_DIR}/run-info.txt"
  ;;
esac

# 额外：缺失 LangSegment 则自动安装（用于 TTS g2p）
if ! "$PYTHON_BIN" - >/dev/null 2>&1 <<'PY'
import importlib, sys
sys.exit(0 if importlib.util.find_spec('LangSegment') else 1)
PY
then
  echo "[dep] installing LangSegment for TTS g2p" | tee -a "${LOG_DIR}/run-info.txt"
  "$PYTHON_BIN" -m pip install LangSegment | tee -a "${LOG_DIR}/run-info.txt"
fi

# 额外：缺失常用 g2p 依赖则安装（避免 TTS 链条缺包）
for PKG in pypinyin jieba cn2an g2p_en phonemizer pyopenjtalk pykakasi; do
  if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('${PKG}') else 1)" >/dev/null 2>&1; then
    echo "[dep] installing ${PKG}" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install "${PKG}" | tee -a "${LOG_DIR}/run-info.txt" || true
  fi
done

# 依赖快速自检与修复：核心三方与扩展（缺则安装）
REQ_PKGS=(
  json5 ruamel.yaml easydict huggingface_hub tqdm
  librosa soundfile ffmpeg-python onnxruntime
  encodec vocos
)
for PKG in "${REQ_PKGS[@]}"; do
  MOD="$PKG"
  [ "$PKG" = "ruamel.yaml" ] && MOD="ruamel"
  [ "$PKG" = "ffmpeg-python" ] && MOD="ffmpeg"
  if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('${MOD}') else 1)" >/dev/null 2>&1; then
    echo "[dep] installing ${PKG}" | tee -a "${LOG_DIR}/run-info.txt"
    "$PYTHON_BIN" -m pip install "${PKG}" | tee -a "${LOG_DIR}/run-info.txt" || true
  fi
done

# 对齐 torchvision 版本以匹配 torch（避免经典不兼容）
"$PYTHON_BIN" - <<'PY'
import re, subprocess, sys
import importlib
def vtuple(v):
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
try:
    import torch
    tv = getattr(torch, '__version__', '0.0.0')
    tmaj, tmin = vtuple(tv)[:2]
    import importlib.util
    need = None
    if tmaj==2 and tmin>=5:
        # torch>=2.5 建议 torchvision 0.20.x
        need = '0.20.1'
    if need:
        ok = False
        if importlib.util.find_spec('torchvision'):
            import torchvision
            ok = vtuple(torchvision.__version__)[0:2] == (0,20)
        if not ok:
            print('[dep] fixing torchvision to', need)
            subprocess.run([sys.executable,'-m','pip','install',f'torchvision=={need}'], check=False)
            import importlib as _il; _il.invalidate_caches()
except Exception as e:
    print('[dep] torchvision check skipped:', e)
PY

# 若 monotonic_align 扩展不可用则自动编译安装（Cython）
if ! "$PYTHON_BIN" -c "import importlib,sys; sys.exit(0 if importlib.util.find_spec('monotonic_align.core') else 1)" >/dev/null 2>&1; then
  echo "[dep] building monotonic_align" | tee -a "${LOG_DIR}/run-info.txt"
  (
    cd "${CODE_DIR}/modules/monotonic_align" && \
    mkdir -p monotonic_align && [ -f monotonic_align/__init__.py ] || echo "" > monotonic_align/__init__.py && \
    "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel cython && \
    "$PYTHON_BIN" setup.py build_ext --inplace && \
    "$PYTHON_BIN" -m pip install -e .
  ) || echo "[dep] monotonic_align build failed (will continue)" | tee -a "${LOG_DIR}/run-info.txt"
fi

#############################
# ===== 预下载模型权重  =====
#############################
# 首次运行建议开启（默认 1）；设置 PREFETCH=0 可跳过
PREFETCH=${PREFETCH:-1}
if [ "$PREFETCH" = "1" ]; then
  echo "==== PREFETCH CKPTS ====" | tee -a "${LOG_DIR}/run-info.txt"
  cd "${CODE_DIR}"
  NEED=0
  [ -d "./ckpts/Vevo/tokenizer/vq8192" ] || NEED=1
  [ -d "./ckpts/Vevo/acoustic_modeling/Vocoder" ] || NEED=1
  if [ "$NEED" = "1" ]; then
    "$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
cache_dir = "./ckpts/Vevo"
patterns = [
    "tokenizer/vq8192/*",
    "tokenizer/vq32/*",
    "contentstyle_modeling/Vq32ToVq8192/*",
    "contentstyle_modeling/PhoneToVq8192/*",
    "acoustic_modeling/Vq8192ToMels/*",
    "acoustic_modeling/Vocoder/*",
]
snapshot_download(repo_id="amphion/Vevo", repo_type="model",
                  cache_dir=cache_dir, allow_patterns=patterns)
print("[prefetch] All Vevo ckpts ready in", cache_dir)
PY
  else
    echo "[prefetch] ckpts already present, skipping" | tee -a "${LOG_DIR}/run-info.txt"
  fi
fi

#############################
# ===== 运行前信息快照  =====
#############################
echo "==== RUN INFO ====" | tee -a "${LOG_DIR}/run-info.txt"
echo "JOB_ID        : ${SLURM_JOB_ID}"        | tee -a "${LOG_DIR}/run-info.txt"
echo "PARTITION     : ${SLURM_JOB_PARTITION}" | tee -a "${LOG_DIR}/run-info.txt"
echo "RUN_DIR       : ${RUN_DIR}"             | tee -a "${LOG_DIR}/run-info.txt"
echo "CODE_DIR      : ${CODE_DIR}"            | tee -a "${LOG_DIR}/run-info.txt"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | tee -a "${LOG_DIR}/run-info.txt"

#############################
# ===== 优雅退出与统计  =====
#############################
record_stats() {
  my-job-stats -a -n -s -j "${SLURM_JOB_ID}" > "${STAT_DIR}/${SLURM_JOB_ID}.stats" 2>&1 || true
  echo "==== my-job-stats summary ====" | tee -a "${LOG_DIR}/run-info.txt"
  tail -n +1 "${STAT_DIR}/${SLURM_JOB_ID}.stats" | tee -a "${LOG_DIR}/run-info.txt" || true
}
trap record_stats EXIT
trap "echo 'Got SIGTERM, wrapping up…' | tee -a '${LOG_DIR}/run-info.txt'; record_stats; exit 1" TERM

#############################
# ===== 切换到代码目录  =====
#############################
cd "${CODE_DIR}"

#############################
# ======= 执行 VEVO  =======
#############################
WAV_DIR="models/vc/vevo/wav"
mkdir -p "$WAV_DIR"

# 如缺少示例输入音频，则自动生成占位测试音频，避免 FileNotFoundError
if [ ! -f "${WAV_DIR}/source.wav" ] || [ ! -f "${WAV_DIR}/arabic_male.wav" ] || [ ! -f "${WAV_DIR}/mandarin_female.wav" ]; then
  "$PYTHON_BIN" - <<'PY'
import numpy as np, soundfile as sf, os
base=os.path.join('models','vc','vevo','wav')
os.makedirs(base, exist_ok=True)
sr=24000
t=np.linspace(0,0.6,int(sr*0.6),endpoint=False)
sig=np.sin(2*np.pi*440*t).astype(np.float32)
sf.write(os.path.join(base,'source.wav'),sig,sr)
sf.write(os.path.join(base,'arabic_male.wav'),sig*0.8,sr)
sf.write(os.path.join(base,'mandarin_female.wav'),sig*0.6,sr)
print('Prepared placeholder wavs in', base)
PY
fi

for mode in timbre style voice tts; do
  echo "==== Running VEVO_MODE=$mode ====" | tee -a "${LOG_DIR}/run-info.txt"

  # 仅清理旧输出，保留示例与输入，避免删除必要输入导致 FileNotFoundError
  rm -f "${WAV_DIR}"/output_*.wav 2>/dev/null || true

  case "$mode" in
    timbre) "$PYTHON_BIN" -m models.vc.vevo.infer_vevotimbre ;;
    style)  "$PYTHON_BIN" -m models.vc.vevo.infer_vevostyle  ;;
    voice)  "$PYTHON_BIN" -m models.vc.vevo.infer_vevovoice  ;;
    tts)    "$PYTHON_BIN" -m models.vc.vevo.infer_vevotts    ;;  # 若缺系统依赖可能不会产出
  esac

  # 收集产物到各自目录
  if ls "${WAV_DIR}"/*.wav >/dev/null 2>&1; then
    rsync -ah --ignore-existing "${WAV_DIR}/" "${OUT_DIR}/${mode}/"
  else
    echo "No WAV produced for mode=${mode} (check Slurm & run-info logs)" | tee -a "${LOG_DIR}/run-info.txt"
  fi
done

echo "==== DONE. Outputs in: ${OUT_DIR} ====" | tee -a "${LOG_DIR}/run-info.txt"
