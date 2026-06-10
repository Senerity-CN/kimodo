#!/usr/bin/env bash
# Batch generate SOMA motions for SONIC fine-tuning.
# Prerequisites: text encoder service running (kimodo_textencoder).
# Usage: cd /home/balance/kimodo && source .venv/bin/activate && bash generate_finetune_motions.sh

set -euo pipefail

export TEXT_ENCODERS_DIR=/home/balance/kimodo/text_encoders
export HF_ENDPOINT=https://hf-mirror.com

MODEL="Kimodo-SOMA-RP-v1.1"
DURATION=5.0
BASE_DIR="outputs"

generate() {
    local category="$1"
    local name="$2"
    local prompt="$3"
    local dur="${4:-$DURATION}"
    local outpath="${BASE_DIR}/${category}/${name}"

    if [[ -f "${outpath}.npz" ]]; then
        echo "[SKIP] ${outpath}.npz already exists"
        return
    fi

    echo "[GEN] ${category}/${name} (${dur}s): ${prompt}"
    kimodo_gen "${prompt}" \
        --model "${MODEL}" \
        --duration "${dur}" \
        --output "${outpath}" \
        --bvh
}

# ============================================================
# Category 1: backward_walking (后退动作)
# ============================================================
echo "========== Backward Walking =========="

generate backward_walking bw_slow_01        "A person walks backward slowly"
generate backward_walking bw_slow_02        "A person slowly steps backward"
generate backward_walking bw_slow_03        "A person carefully walks backward step by step"
generate backward_walking bw_slow_04        "A person takes slow steps backward"
generate backward_walking bw_slow_05        "A person moves backward at a slow pace"

generate backward_walking bw_normal_01      "A person walks backward"
generate backward_walking bw_normal_02      "A person steps backward steadily"
generate backward_walking bw_normal_03      "A person walks backward with a steady pace"
generate backward_walking bw_normal_04      "A person moves backward naturally"
generate backward_walking bw_normal_05      "A person backs up while walking"

generate backward_walking bw_fast_01        "A person walks backward quickly"
generate backward_walking bw_fast_02        "A person quickly steps backward"
generate backward_walking bw_fast_03        "A person jogs backward"
generate backward_walking bw_fast_04        "A person rapidly walks backward"
generate backward_walking bw_fast_05        "A person hurries backward"

generate backward_walking bw_turn_01        "A person walks backward and turns left"
generate backward_walking bw_turn_02        "A person walks backward and turns right"
generate backward_walking bw_turn_03        "A person steps backward while turning"
generate backward_walking bw_turn_04        "A person walks backward in a curve"
generate backward_walking bw_turn_05        "A person backs up and changes direction"

generate backward_walking bw_look_01        "A person walks backward while looking over their shoulder"
generate backward_walking bw_look_02        "A person cautiously walks backward"
generate backward_walking bw_look_03        "A person retreats backward carefully"
generate backward_walking bw_look_04        "A person backs away slowly with caution"
generate backward_walking bw_look_05        "A person steps backward nervously"

# ============================================================
# Category 2: small_steps (小碎步 / 微调步态)
# ============================================================
echo "========== Small Steps =========="

generate small_steps ss_shuffle_01          "A person shuffles in place"
generate small_steps ss_shuffle_02          "A person shuffles their feet in place"
generate small_steps ss_shuffle_03          "A person does a standing shuffle"
generate small_steps ss_shuffle_04          "A person shifts weight from foot to foot"
generate small_steps ss_shuffle_05          "A person fidgets while standing"

generate small_steps ss_small_fwd_01        "A person takes very small steps forward"
generate small_steps ss_small_fwd_02        "A person inches forward slowly"
generate small_steps ss_small_fwd_03        "A person walks forward with tiny steps"
generate small_steps ss_small_fwd_04        "A person creeps forward cautiously"
generate small_steps ss_small_fwd_05        "A person tiptoes forward"

generate small_steps ss_lateral_01          "A person sidesteps to the left slowly"
generate small_steps ss_lateral_02          "A person sidesteps to the right slowly"
generate small_steps ss_lateral_03          "A person takes small lateral steps"
generate small_steps ss_lateral_04          "A person shuffles sideways to the left"
generate small_steps ss_lateral_05          "A person shuffles sideways to the right"

generate small_steps ss_adjust_01           "A person adjusts their footing"
generate small_steps ss_adjust_02           "A person makes small stepping adjustments"
generate small_steps ss_adjust_03           "A person repositions with small steps"
generate small_steps ss_adjust_04           "A person takes a half step forward then stops"
generate small_steps ss_adjust_05           "A person takes one small step to the side"

generate small_steps ss_slow_walk_01        "A person walks very slowly"
generate small_steps ss_slow_walk_02        "A person walks slowly and cautiously"
generate small_steps ss_slow_walk_03        "A person walks with a slow careful gait"
generate small_steps ss_slow_walk_04        "A person strolls at a very slow pace"
generate small_steps ss_slow_walk_05        "A person walks slowly looking at the ground"

# ============================================================
# Category 3: upper_body_reach (上肢伸展 / 传递)
# ============================================================
echo "========== Upper Body Reach =========="

generate upper_body_reach ub_reach_fwd_01   "A person reaches forward with both hands"
generate upper_body_reach ub_reach_fwd_02   "A person extends both arms forward"
generate upper_body_reach ub_reach_fwd_03   "A person stretches arms forward to grab something"
generate upper_body_reach ub_reach_fwd_04   "A person pushes both hands forward"
generate upper_body_reach ub_reach_fwd_05   "A person reaches out with both arms"

generate upper_body_reach ub_reach_one_01   "A person reaches forward with right hand"
generate upper_body_reach ub_reach_one_02   "A person reaches forward with left hand"
generate upper_body_reach ub_reach_one_03   "A person extends right arm to pick something up"
generate upper_body_reach ub_reach_one_04   "A person extends left arm to the side"
generate upper_body_reach ub_reach_one_05   "A person points forward with right arm extended"

generate upper_body_reach ub_raise_01       "A person raises both arms above their head"
generate upper_body_reach ub_raise_02       "A person lifts arms overhead"
generate upper_body_reach ub_raise_03       "A person raises hands up high"
generate upper_body_reach ub_raise_04       "A person stretches both arms upward"
generate upper_body_reach ub_raise_05       "A person waves both hands above their head"

generate upper_body_reach ub_pass_01        "A person passes an object with both hands"
generate upper_body_reach ub_pass_02        "A person hands something over with both hands"
generate upper_body_reach ub_pass_03        "A person holds out an object at chest height"
generate upper_body_reach ub_pass_04        "A person offers something with outstretched arms"
generate upper_body_reach ub_pass_05        "A person gives a box to someone in front"

generate upper_body_reach ub_lateral_01     "A person stretches arms out to the sides"
generate upper_body_reach ub_lateral_02     "A person spreads arms wide open"
generate upper_body_reach ub_lateral_03     "A person reaches to the right with right arm"
generate upper_body_reach ub_lateral_04     "A person reaches to the left with left arm"
generate upper_body_reach ub_lateral_05     "A person rotates torso and reaches to the side"

echo ""
echo "========== Generation Complete =========="
echo "Backward walking: $(ls ${BASE_DIR}/backward_walking/*.npz 2>/dev/null | wc -l) motions"
echo "Small steps:      $(ls ${BASE_DIR}/small_steps/*.npz 2>/dev/null | wc -l) motions"
echo "Upper body reach: $(ls ${BASE_DIR}/upper_body_reach/*.npz 2>/dev/null | wc -l) motions"
