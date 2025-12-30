#!/bin/bash

# --- ensure micromamba env; if activation fails, run env.sh then retry ---
ensure_env() {
    # load micromamba shell hook if present
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)" || true
    fi

    # try to activate; on failure, run env.sh and retry once
    if ! micromamba activate com4d >/dev/null 2>&1; then
        echo "[setup] activation failed for 'com4d'. Running env.sh then retrying ..."
        bash scripts/env.sh || { echo "[setup] env.sh failed"; exit 1; }
        eval "$(micromamba shell hook -s bash)" || true
        micromamba activate com4d || { echo "[setup] activation still failing"; exit 1; }
    fi
}

ensure_env

micromamba activate com4d

python src/infer_com4d.py \
    --tag demo2 \
    --frames_dir FRAMES_DIR \
    --masks_dir MASKS_DIR \
    --output_dir ./com4dout \
    --base_weights_dir pretrained_weights/TripoSG \
    --transformer_dir transformer_ema \
    --num_tokens 1024 \
    --frames_start_idx 0 \
    --frames_end_idx 100 \
    --frame_stride 2 \
    --scene_num_parts 2 \
    --dynamic_num_parts 1 \
    --scene_block_size 16 \
    --dynamic_block_size 6 \
    --dynamic_max_memory_frames 8 \
    --animation \
    --animation_insert_rotation_every 10 \
    --scene_mix_cutoff 50 \
    --dynamic_mix_cutoff 10 \


