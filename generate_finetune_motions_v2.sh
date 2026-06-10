#!/usr/bin/env bash
# Supplementary motion generation for SONIC fine-tuning (v2).
# Focus: chair-pulling backward + menu handover, plus general diversity.
# Prerequisites: text encoder service running (kimodo_textencoder).
# Usage: cd /home/balance/kimodo && source .venv/bin/activate && bash generate_finetune_motions_v2.sh

set -euo pipefail

export TEXT_ENCODERS_DIR=/home/balance/kimodo/text_encoders
export HF_ENDPOINT=https://hf-mirror.com

MODEL="Kimodo-SOMA-RP-v1.1"
BASE_DIR="outputs"

generate() {
    local category="$1"
    local name="$2"
    local prompt="$3"
    local dur="${4:-8.0}"
    local nsamples="${5:-1}"
    local outpath="${BASE_DIR}/${category}/${name}"

    if [[ "${nsamples}" -eq 1 ]]; then
        if [[ -f "${outpath}.npz" ]]; then
            echo "[SKIP] ${outpath}.npz already exists"
            return
        fi
    else
        if [[ -f "${outpath}/${name}_00.npz" ]]; then
            echo "[SKIP] ${outpath}/ already exists (multi-sample)"
            return
        fi
    fi

    echo "[GEN] ${category}/${name} (${dur}s, x${nsamples}): ${prompt}"
    kimodo_gen "${prompt}" \
        --model "${MODEL}" \
        --duration "${dur}" \
        --num_samples "${nsamples}" \
        --output "${outpath}" \
        --bvh
}

# ============================================================
# Category 4: chair_pull (拖椅子后退) — 40 clips, 8-10s
# 核心场景：弯腰抓住椅子 → 后退拖拽 → 停下松手
# ============================================================
echo "========== Chair Pull Backward =========="

# 4a. 弯腰抓+直接后拖 (基础动作) — 核心 prompt x3 samples
generate chair_pull cp_basic_01    "A person bends forward, grabs a chair with both hands, and pulls it backward" 8.0 3
generate chair_pull cp_basic_02    "A person reaches down, grips a chair back, and drags it backward slowly" 8.0 3
generate chair_pull cp_basic_03    "A person pulls a chair backward with both hands while stepping back" 8.0 3
generate chair_pull cp_basic_04    "A person grabs a chair and walks backward dragging it" 8.0
generate chair_pull cp_basic_05    "A person leans forward to grab a chair and pulls it back a few steps" 8.0

# 4b. 慢速小幅拖拽 (精细控制) — 核心 prompt x3
generate chair_pull cp_slow_01     "A person slowly pulls a chair backward one step at a time" 8.0 3
generate chair_pull cp_slow_02     "A person carefully drags a chair backward with small steps" 8.0 3
generate chair_pull cp_slow_03     "A person gently pulls a heavy object backward step by step" 10.0
generate chair_pull cp_slow_04     "A person inches backward while pulling something with both hands" 8.0
generate chair_pull cp_slow_05     "A person takes two small steps backward while holding onto something" 8.0

# 4c. 弯腰拖拽姿态 (保持前倾) — 核心 prompt x3
generate chair_pull cp_bent_01     "A person leans forward and walks backward pulling a heavy object" 8.0 3
generate chair_pull cp_bent_02     "A person bends at the waist, grabs something low, and drags it backward" 8.0 3
generate chair_pull cp_bent_03     "A person crouches slightly and pulls something backward with effort" 8.0
generate chair_pull cp_bent_04     "A person stoops forward to drag an object backward across the floor" 10.0
generate chair_pull cp_bent_05     "A person bends down and drags a box backward" 8.0

# 4d. 拖拽后松手/站直 (完整流程) — 核心 prompt x3
generate chair_pull cp_release_01  "A person pulls something backward then stands up straight" 10.0
generate chair_pull cp_release_02  "A person drags a chair back two steps then lets go and stands upright" 10.0 3
generate chair_pull cp_release_03  "A person walks backward while pulling, then stops and releases" 10.0 3
generate chair_pull cp_release_04  "A person pulls a heavy object backward and then straightens up" 10.0
generate chair_pull cp_release_05  "A person drags something backward, stops, and steps to the side" 10.0

# 4e. 单手拖拽 (变体)
generate chair_pull cp_one_01      "A person grabs a chair with one hand and pulls it backward" 8.0
generate chair_pull cp_one_02      "A person reaches back with one hand to drag a chair while stepping backward" 8.0
generate chair_pull cp_one_03      "A person pulls something backward with the right hand" 8.0
generate chair_pull cp_one_04      "A person drags an object backward using the left hand" 8.0
generate chair_pull cp_one_05      "A person steps backward while pulling something with one arm" 8.0

# 4f. 侧向/对角拖拽
generate chair_pull cp_angle_01    "A person pulls a chair backward and to the left" 8.0
generate chair_pull cp_angle_02    "A person pulls a chair backward and to the right" 8.0
generate chair_pull cp_angle_03    "A person drags something backward at an angle while turning" 8.0
generate chair_pull cp_angle_04    "A person steps backward diagonally while pulling an object" 8.0
generate chair_pull cp_angle_05    "A person walks backward in a curve while dragging something" 8.0

# 4g. 用力拖拽 (重物感) — 核心 prompt x3
generate chair_pull cp_heavy_01    "A person pulls a heavy object backward with effort using both hands" 8.0 3
generate chair_pull cp_heavy_02    "A person struggles to drag a heavy piece of furniture backward" 10.0 3
generate chair_pull cp_heavy_03    "A person leans back and pulls hard to drag something backward" 8.0
generate chair_pull cp_heavy_04    "A person uses body weight to pull a heavy object backward" 8.0
generate chair_pull cp_heavy_05    "A person braces and drags a heavy box backward slowly" 8.0

# 4h. 整理动线 (拖完→让位) — 核心 prompt x3
generate chair_pull cp_aside_01    "A person pulls a chair backward then steps aside to make room" 10.0 3
generate chair_pull cp_aside_02    "A person drags a chair back and gestures for someone to sit" 10.0 3
generate chair_pull cp_aside_03    "A person pulls a chair out, steps to the side, and extends an arm inviting" 10.0
generate chair_pull cp_aside_04    "A person moves a chair backward and stands beside it" 10.0
generate chair_pull cp_aside_05    "A person pulls a chair back and holds it steady with one hand" 10.0

# ============================================================
# Category 5: menu_handover (递菜单/递物) — 30 clips, 8s
# 核心场景：拿起菜单 → 前伸递出 → 稳定持住等对方接
# ============================================================
echo "========== Menu Handover =========="

# 5a. 双手前递 (标准递法) — 核心 prompt x3 samples
generate menu_handover mh_both_01  "A person holds a flat object with both hands and presents it forward" 8.0 3
generate menu_handover mh_both_02  "A person extends a book forward with both hands at chest height" 8.0 3
generate menu_handover mh_both_03  "A person offers a tray to someone in front with both hands" 8.0 3
generate menu_handover mh_both_04  "A person holds out a clipboard with both hands to someone" 8.0
generate menu_handover mh_both_05  "A person presents a folder forward with both hands politely" 8.0

# 5b. 单手前递
generate menu_handover mh_one_01   "A person extends the right hand forward to hand over a flat object" 8.0
generate menu_handover mh_one_02   "A person holds out a book with one hand to someone" 8.0
generate menu_handover mh_one_03   "A person offers something with the right arm extended forward" 8.0
generate menu_handover mh_one_04   "A person reaches forward with the left hand to give something" 8.0
generate menu_handover mh_one_05   "A person hands an object forward with one hand at waist height" 8.0

# 5c. 拿起→递出 (完整流程) — 核心 prompt x3
generate menu_handover mh_pick_01  "A person picks up a book from a table and hands it to someone" 8.0 3
generate menu_handover mh_pick_02  "A person grabs a folder from the side and presents it forward" 8.0 3
generate menu_handover mh_pick_03  "A person reaches to the side, picks something up, and extends it forward" 10.0
generate menu_handover mh_pick_04  "A person bends down, picks up an object, and holds it out" 10.0
generate menu_handover mh_pick_05  "A person lifts something from a surface and offers it with both hands" 8.0

# 5d. 前倾递物 (带身体前倾) — 核心 prompt x3
generate menu_handover mh_lean_01  "A person leans forward slightly to hand something to another person" 8.0 3
generate menu_handover mh_lean_02  "A person bends forward at the waist to offer something politely" 8.0 3
generate menu_handover mh_lean_03  "A person steps forward and extends both arms to give a flat object" 8.0
generate menu_handover mh_lean_04  "A person takes a step and leans forward to present a book" 8.0
generate menu_handover mh_lean_05  "A person walks one step forward and reaches out to hand something over" 8.0

# 5e. 递到不同高度/方向 — 递给坐着的人 x3
generate menu_handover mh_dir_01   "A person holds out an object to the left side" 8.0
generate menu_handover mh_dir_02   "A person holds out an object to the right side" 8.0
generate menu_handover mh_dir_03   "A person extends arms downward to hand something to a seated person" 8.0 3
generate menu_handover mh_dir_04   "A person reaches slightly to the side to offer something" 8.0
generate menu_handover mh_dir_05   "A person holds a flat object at different heights presenting it" 8.0

# 5f. 递完收手/等待
generate menu_handover mh_wait_01  "A person holds out a book and waits patiently" 8.0
generate menu_handover mh_wait_02  "A person presents an object and holds the pose steadily" 8.0
generate menu_handover mh_wait_03  "A person offers something forward then slowly pulls hands back" 8.0
generate menu_handover mh_wait_04  "A person hands over an object and steps back politely" 10.0
generate menu_handover mh_wait_05  "A person gives something with both hands then returns to standing" 10.0

# ============================================================
# Category 6: backward_walking_ext (后退补充) — 20 clips, 8s
# ============================================================
echo "========== Backward Walking Extended =========="

# 6a. 后退+上肢动作
generate backward_ext bwe_arm_01   "A person walks backward while holding arms forward" 8.0
generate backward_ext bwe_arm_02   "A person steps backward with hands raised to chest" 8.0
generate backward_ext bwe_arm_03   "A person backs up while waving hands" 8.0
generate backward_ext bwe_arm_04   "A person walks backward while reaching forward with one hand" 8.0
generate backward_ext bwe_arm_05   "A person retreats backward while gesturing with both arms" 8.0

# 6b. 后退对角线/弧线
generate backward_ext bwe_diag_01  "A person walks backward diagonally to the left" 8.0
generate backward_ext bwe_diag_02  "A person walks backward diagonally to the right" 8.0
generate backward_ext bwe_diag_03  "A person backs up in a wide arc" 8.0
generate backward_ext bwe_diag_04  "A person walks backward while drifting to one side" 8.0
generate backward_ext bwe_diag_05  "A person takes several steps backward in a zigzag" 8.0

# 6c. 后退启停过渡
generate backward_ext bwe_stop_01  "A person takes two steps backward then stops" 8.0
generate backward_ext bwe_stop_02  "A person backs up, pauses, then backs up more" 10.0
generate backward_ext bwe_stop_03  "A person walks backward and gradually slows to a stop" 8.0
generate backward_ext bwe_stop_04  "A person starts walking backward from a standstill" 8.0
generate backward_ext bwe_stop_05  "A person hesitates, then takes a few steps backward" 8.0

# 6d. 后退+看/转头
generate backward_ext bwe_look_01  "A person walks backward while turning head to look behind" 8.0
generate backward_ext bwe_look_02  "A person backs up while looking over the right shoulder" 8.0
generate backward_ext bwe_look_03  "A person steps backward carefully glancing behind" 8.0
generate backward_ext bwe_look_04  "A person retreats backward while scanning the area" 8.0
generate backward_ext bwe_look_05  "A person walks backward with head turned to the side" 8.0

# ============================================================
# Category 7: small_steps_ext (碎步补充) — 20 clips, 8s
# ============================================================
echo "========== Small Steps Extended =========="

# 7a. 原地转体
generate small_steps_ext sse_pivot_01  "A person pivots in place turning to the left" 8.0
generate small_steps_ext sse_pivot_02  "A person pivots in place turning to the right" 8.0
generate small_steps_ext sse_pivot_03  "A person turns around slowly in place with small steps" 8.0
generate small_steps_ext sse_pivot_04  "A person rotates body to face another direction using tiny steps" 8.0
generate small_steps_ext sse_pivot_05  "A person does a slow quarter turn in place" 8.0

# 7b. 重心转移/站姿微调
generate small_steps_ext sse_shift_01  "A person shifts weight from the left foot to the right foot" 8.0
generate small_steps_ext sse_shift_02  "A person sways gently from side to side while standing" 8.0
generate small_steps_ext sse_shift_03  "A person rocks back and forth on their feet" 8.0
generate small_steps_ext sse_shift_04  "A person lifts one foot slightly then puts it back down" 8.0
generate small_steps_ext sse_shift_05  "A person shifts stance and adjusts balance" 8.0

# 7c. 启停模式
generate small_steps_ext sse_startstop_01  "A person takes one step forward then stops" 8.0
generate small_steps_ext sse_startstop_02  "A person starts walking then immediately stops" 8.0
generate small_steps_ext sse_startstop_03  "A person takes two steps then pauses and adjusts footing" 8.0
generate small_steps_ext sse_startstop_04  "A person walks a short distance then stops abruptly" 8.0
generate small_steps_ext sse_startstop_05  "A person moves forward one step, backward one step" 8.0

# 7d. 碎步+手部动作
generate small_steps_ext sse_hand_01  "A person shuffles forward while holding something" 8.0
generate small_steps_ext sse_hand_02  "A person takes small steps while reaching forward" 8.0
generate small_steps_ext sse_hand_03  "A person tiptoes forward with arms extended" 8.0
generate small_steps_ext sse_hand_04  "A person makes small adjusting steps while holding arms up" 8.0
generate small_steps_ext sse_hand_05  "A person sidesteps slowly while carrying something" 8.0

# ============================================================
# Category 8: combined (组合动作) — 25 clips, 10s
# ============================================================
echo "========== Combined Motions =========="

# 8a. 后退+递物 (核心组合)
generate combined cb_back_hand_01  "A person walks backward while holding out an object with both hands" 10.0
generate combined cb_back_hand_02  "A person steps backward and extends arms to offer something" 10.0
generate combined cb_back_hand_03  "A person backs up while presenting a tray forward" 10.0
generate combined cb_back_hand_04  "A person retreats slowly while handing something over" 10.0
generate combined cb_back_hand_05  "A person walks backward while keeping arms extended forward" 10.0

# 8b. 慢走+递物
generate combined cb_walk_hand_01  "A person walks slowly forward and hands something to another person" 10.0
generate combined cb_walk_hand_02  "A person approaches slowly and offers a book with both hands" 10.0
generate combined cb_walk_hand_03  "A person walks up to someone and presents a flat object" 10.0
generate combined cb_walk_hand_04  "A person steps forward carefully and extends an object" 10.0
generate combined cb_walk_hand_05  "A person walks slowly and offers something at chest height" 10.0

# 8c. 弯腰+后退 (拖拽类)
generate combined cb_bend_back_01  "A person bends down, grabs an object, and walks backward" 10.0
generate combined cb_bend_back_02  "A person squats to grip something then steps backward pulling it" 10.0
generate combined cb_bend_back_03  "A person leans forward and shuffles backward while holding on" 10.0
generate combined cb_bend_back_04  "A person bends at the waist and takes careful steps backward" 10.0
generate combined cb_bend_back_05  "A person stoops and drags something backward with small steps" 10.0

# 8d. 碎步+转身+伸手
generate combined cb_shuffle_reach_01  "A person shuffles sideways then reaches out with one arm" 10.0
generate combined cb_shuffle_reach_02  "A person adjusts position with small steps and extends both arms" 10.0
generate combined cb_shuffle_reach_03  "A person turns in place with small steps then reaches forward" 10.0
generate combined cb_shuffle_reach_04  "A person sidesteps and presents something to the side" 10.0
generate combined cb_shuffle_reach_05  "A person pivots with tiny steps and offers an object" 10.0

# 8e. 服务动线：走近→递物→后退 — 核心组合 x3
generate combined cb_serve_01      "A person walks forward, extends an object, then steps backward" 10.0 3
generate combined cb_serve_02      "A person approaches, presents something with both hands, then retreats" 10.0 3
generate combined cb_serve_03      "A person walks up, leans forward to give something, then backs away" 10.0 3
generate combined cb_serve_04      "A person steps forward to hand over a folder, then steps back politely" 10.0
generate combined cb_serve_05      "A person offers a tray to someone, then takes a few steps backward" 10.0

V2_DIRS="${BASE_DIR}/chair_pull ${BASE_DIR}/menu_handover ${BASE_DIR}/backward_ext ${BASE_DIR}/small_steps_ext ${BASE_DIR}/combined"

echo ""
echo "========== V2 Generation Complete =========="
echo "Chair pull:         $(find ${BASE_DIR}/chair_pull -name '*.npz' 2>/dev/null | wc -l) clips"
echo "Menu handover:      $(find ${BASE_DIR}/menu_handover -name '*.npz' 2>/dev/null | wc -l) clips"
echo "Backward ext:       $(find ${BASE_DIR}/backward_ext -name '*.npz' 2>/dev/null | wc -l) clips"
echo "Small steps ext:    $(find ${BASE_DIR}/small_steps_ext -name '*.npz' 2>/dev/null | wc -l) clips"
echo "Combined:           $(find ${BASE_DIR}/combined -name '*.npz' 2>/dev/null | wc -l) clips"
V2_COUNT=$(find ${V2_DIRS} -name '*.npz' 2>/dev/null | wc -l)
echo ""
echo "V1 existing:        75 clips"
echo "V2 new:             ${V2_COUNT} clips  (24 prompts x3 samples + 111 prompts x1)"
echo "Total:              $((75 + V2_COUNT)) clips"
