#!/bin/bash
###############################################################################
#  文件名: vevo_run.slurm
#  用途  : 在 Spartan 上运行 Amphion/VEVO 的推理（一次跑四种模式）
#  说明  : 含常见 Slurm 参数、Conda 激活、运行归档(runs/)、my-job-stats 统计
###############################################################################

#############################
# ===== Slurm 资源申请 =====
#############################
#SBATCH --partition=gpu-a100-short       # 短作业调试分区（≤4h）；长任务改为 gpu-a100 / gpu-h100
#SBATCH --gres=gpu:1                     # 1 块 GPU 足够跑推理 demo
#SBATCH -c 4                             # 4 个 CPU 核用于解码/数据加载
#SBATCH --mem=16G                        # 16GB 内存（推理足够）
#SBATCH -t 01:00:00                      # Walltime 1 小时（demo 几分钟即可）
#SBATCH -J vevo-infer                    # 作业名
#SBATCH -o logs/%x-%j.out                # 标准输出日志
#SBATCH -e logs/%x-%j.err                # 标准错误日志
#SBATCH --signal=TERM@60                 # 到时限前 60 秒发 SIGTERM，便于收尾

# 邮件通知
#SBATCH --mail-user=haoguangz@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL

# 项目账号（确保记账到你的项目）
#SBATCH -A punim2341

#############################
# ===== 安全/健壮设置  =====
#############################
set -euo pipefail   # -e: 任一命令失败即退出；-u: 未定义变量即报错；-o pipefail: 管道失败即失败

#############################
# ===== 路径与环境变量  =====
#############################
PROJECT_ROOT=/data/gpfs/projects/punim2341/haoguangzhou
CODE_DIR="${PROJECT_ROOT}/voice/Amphion-VevoDev"
RUNS_ROOT="${PROJECT_ROOT}/runs"
CONDA_ENV=vevo

# 模型/权重缓存放项目目录，避免撑爆 $HOME
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PROJECT_ROOT}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

# 节点本地临时目录（系统提供）
export TMPDIR=${TMPDIR:-/tmp}

#############################
# ===== 生成本次运行目录 =====
#############################
timestamp=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RUNS_ROOT}/${timestamp}_vevo_infer"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
STAT_DIR="${RUN_DIR}/stats"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${STAT_DIR}"

#############################
# ===== 加载与激活环境  =====
#############################
module purge
module load Anaconda3/2024.02-1           # 仅加载 Anaconda 模块（不要混用其它 DL 模块）
eval "$(conda shell.bash hook)"           # 启用 conda 命令（不要用 conda init）
conda activate "${CONDA_ENV}"             # 激活你已安装依赖的环境

#############################
# ===== 运行前信息快照  =====
#############################
echo "==== RUN INFO ====" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
echo "JOB_ID        : ${SLURM_JOB_ID}"        | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
echo "PARTITION     : ${SLURM_JOB_PARTITION}" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
echo "RUN_DIR       : ${RUN_DIR}"             | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
echo "CODE_DIR      : ${CODE_DIR}"            | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"

#############################
# ===== 优雅退出与统计  =====
#############################
record_stats() {
  echo "==== my-job-stats @ $(date) ====" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
  my-job-stats -a -n -s -j "${SLURM_JOB_ID}" > "${STAT_DIR}/${SLURM_JOB_ID}.stats" 2>&1 || true
  tail -n +1 "${STAT_DIR}/${SLURM_JOB_ID}.stats" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out" || true
}
trap record_stats EXIT
trap "echo 'Got SIGTERM, wrapping up...'; record_stats; exit 1" TERM

#############################
# ===== 切换到代码目录  =====
#############################
cd "${CODE_DIR}"

#############################
# ======= 执行 VEVO  =======
#############################
# 逐个模式运行（官方 demo 会把 wav 写到 models/vc/vevo/wav/）
WAV_DIR="models/vc/vevo/wav"
mkdir -p "$WAV_DIR"

for mode in timbre style voice tts; do
  echo "==== Running VEVO_MODE=$mode ====" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"

  # 清空旧输出，避免混入上一轮的 wav
  rm -f "${WAV_DIR}"/*.wav 2>/dev/null || true

  case "$mode" in
    timbre) python -m models.vc.vevo.infer_vevotimbre ;;
    style)  python -m models.vc.vevo.infer_vevostyle  ;;
    voice)  python -m models.vc.vevo.infer_vevovoice  ;;
    tts)    python -m models.vc.vevo.infer_vevotts    ;;  # 若缺系统依赖(phonemizer/espeak-ng)可能不会产出
  esac

  # 将当前模式的产物拷入对应目录
  if ls "${WAV_DIR}"/*.wav >/dev/null 2>&1; then
    rsync -ah --ignore-existing "${WAV_DIR}/" "${OUT_DIR}/${mode}/"
  else
    echo "No WAV produced for mode=${mode} (check logs)" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
  fi
done

echo "==== DONE. Outputs in: ${OUT_DIR} ====" | tee -a "${LOG_DIR}/${SLURM_JOB_ID}.out"
