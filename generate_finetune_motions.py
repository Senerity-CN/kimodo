#!/usr/bin/env python3
"""Batch generate low-dynamic motions for SONIC fine-tuning.

Loads the Kimodo model once, then iterates through all prompts generating
num_samples variations per prompt at each duration. Supports resuming.

Prerequisites:
    Text encoder service running: kimodo_textencoder

Usage:
    cd /home/balance/kimodo
    source .venv/bin/activate
    python generate_finetune_motions.py                     # generate all
    python generate_finetune_motions.py --category bend     # only categories matching "bend"
    python generate_finetune_motions.py --dry-run            # list prompts and counts
    python generate_finetune_motions.py --num-samples 5      # 5 variations per prompt
"""

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Motion prompt definitions
# ---------------------------------------------------------------------------
# Each category: {"durations": [float, ...], "prompts": [str, ...]}
#
# Core categories (user-specified) use multiple durations for diversity.
# Each prompt is generated at EVERY listed duration, each with num_samples
# random variations -> prompts x durations x num_samples total data points.
#
# Supplementary categories use a single duration.
# ---------------------------------------------------------------------------

CATEGORIES = OrderedDict()

# ===================================================================
# CORE CATEGORIES — multiple durations for more data
# ===================================================================

# ===== 1. lateral_step_small (小步侧移, 不交叉腿) =====
CATEGORIES["lateral_step_small"] = {
    "durations": [3.0, 5.0, 7.0],
    "prompts": [
        "A person takes a small careful step to the left side and brings their feet together",
        "A person steps sideways to the right with a short controlled step",
        "A person shifts to the left with a small lateral shuffle keeping feet parallel",
        "A person moves to the right side with a single small sidestep",
        "A person takes a gentle sidestep to the left to adjust their position",
        "A person makes a short lateral step to the right while maintaining balance",
        "A person shuffles one small step to the left side",
        "A person carefully sidesteps to the right and returns to a standing pose",
        "A person takes a slow measured step sideways to the left",
        "A person shifts their weight and takes a small step to the right",
        "A person takes a short side step to the left keeping a wide stable stance",
        "A person makes a small precise lateral step to the right",
        "A person eases sideways to the left with a gentle small step",
        "A person slides one step to the right and plants both feet",
        "A person adjusts their position with a small step to the left",
        "A person takes a compact sidestep to the right keeping knees slightly bent",
        "A person slowly shuffles to the left with small deliberate steps",
        "A person makes a quick small sidestep to the right to avoid something",
        "A person takes one controlled step to the left then stands still",
        "A person carefully moves sideways to the right with a half step",
        "A person takes two small shuffle steps to the left",
        "A person sidesteps to the right twice with tiny steps",
        "A person takes a small cautious step to the left while looking forward",
        "A person shifts sideways to the right with a small balanced step",
        "A person slowly scoots to the left with feet close together",
        "A person makes a gentle adjustment step to the right side",
        "A person takes a short step to the left to make room",
        "A person shuffles slightly to the right to reposition",
        "A person takes a measured sidestep to the left keeping their torso upright",
        "A person steps to the right side with a short careful movement",
        "A person takes a small step to the left and pauses in place",
        "A person gently shifts to the right with a tiny lateral step",
        "A person takes a short sidestep to the left while keeping arms relaxed",
        "A person moves a half step to the right maintaining good posture",
        "A person takes three small shuffle steps to the left",
        "A person makes multiple tiny sidesteps to the right",
        "A person carefully inches to the left with small lateral steps",
        "A person scoots to the right with several small shuffling movements",
        "A person takes a small step sideways to the left keeping both feet flat",
        "A person shifts laterally to the right with a gentle controlled step",
    ],
}

# ===== 2. lateral_step_large (大步侧移) =====
CATEGORIES["lateral_step_large"] = {
    "durations": [4.0, 7.0],
    "prompts": [
        "A person takes a wide step to the left side",
        "A person makes a big lateral step to the right",
        "A person lunges sideways to the left with a large step",
        "A person takes a long step to the right side and brings feet together",
        "A person strides to the left with a wide lateral movement",
        "A person takes a large sidestep to the right to dodge something",
        "A person makes a big step to the left while keeping balance",
        "A person takes a wide confident step sideways to the right",
        "A person steps far to the left with a controlled lunge",
        "A person stretches to take a large step to the right side",
        "A person takes an exaggerated sidestep to the left",
        "A person makes a wide reaching step to the right",
        "A person takes a big powerful step to the left side",
        "A person strides sideways to the right with a wide stance",
        "A person takes a large deliberate step to the left and pauses",
        "A person reaches to the right with a wide lateral step",
        "A person takes a long confident step sideways to the left",
        "A person makes a sweeping step to the right side",
        "A person takes two large steps to the left side",
        "A person sidesteps twice to the right with wide strides",
        "A person takes a big step to the left and steadies themselves",
        "A person makes a broad lateral step to the right keeping arms out",
        "A person takes a wide step sideways to the left with arms for balance",
        "A person lunges to the right with a big lateral stride",
        "A person takes a large step to the left then returns to center",
        "A person makes a wide step to the right then stands upright",
        "A person takes a big reaching step to the left side and holds",
        "A person strides broadly to the right in a controlled manner",
        "A person takes a large careful step to the left side",
        "A person steps widely to the right with a slow deliberate motion",
        "A person takes a wide lateral step to the left keeping torso stable",
        "A person makes a large sidestep to the right while looking ahead",
        "A person takes a substantial step to the left",
        "A person moves far to the right with one large step",
        "A person takes a sweeping wide step to the left and plants both feet",
    ],
}

# ===== 3. backward_step_small (小步后退) =====
CATEGORIES["backward_step_small"] = {
    "durations": [3.0, 5.0, 7.0],
    "prompts": [
        "A person takes small careful steps backward slowly",
        "A person inches backward with tiny cautious steps",
        "A person slowly shuffles backward with short steps",
        "A person takes very small steps in reverse",
        "A person carefully backs up with small measured steps",
        "A person retreats backward with gentle small steps",
        "A person steps backward slowly one small step at a time",
        "A person moves backward with tiny deliberate steps",
        "A person backs away with short hesitant steps",
        "A person takes small tentative steps backward while looking forward",
        "A person shuffles backward very slowly with careful footing",
        "A person creeps backward with minimal step size",
        "A person slowly moves in reverse with compact steps",
        "A person takes several tiny steps backward",
        "A person carefully steps backward keeping close to the ground",
        "A person inches back with small controlled steps",
        "A person backs up slowly taking one small step after another",
        "A person walks backward with very short careful strides",
        "A person takes a few small steps backward and stops",
        "A person slowly retreats with small gentle steps",
        "A person cautiously steps backward with tiny movements",
        "A person takes small shuffling steps backward",
        "A person moves backward at a very slow pace with short steps",
        "A person steps back slowly with careful foot placement",
        "A person takes minimal backward steps while maintaining balance",
        "A person shuffles in reverse with small steady steps",
        "A person edges backward with cautious small steps",
        "A person slowly backs away with steady tiny steps",
        "A person takes three small steps backward and pauses",
        "A person gently walks backward with small precise steps",
        "A person moves backward slowly keeping steps close together",
        "A person takes short backward steps while keeping posture upright",
        "A person backs up with controlled small movements",
        "A person takes a couple of small careful steps backward",
        "A person walks backward slowly with a gentle shuffling gait",
        "A person takes small backward steps with arms slightly out for balance",
        "A person creeps backward carefully with short strides",
        "A person takes tiny steps in reverse looking over their shoulder",
        "A person shuffles a few small steps backward",
        "A person carefully moves backward with compact steady steps",
    ],
}

# ===== 4. forward_step_small (小步前走) =====
CATEGORIES["forward_step_small"] = {
    "durations": [3.0, 5.0, 7.0],
    "prompts": [
        "A person takes very small steps forward slowly",
        "A person inches forward with tiny deliberate steps",
        "A person walks forward with very short careful strides",
        "A person creeps forward slowly with small steps",
        "A person takes small cautious steps forward",
        "A person shuffles forward with tiny measured steps",
        "A person moves forward at a slow pace with minimal steps",
        "A person tiptoes forward with small gentle steps",
        "A person takes several tiny steps forward",
        "A person walks forward very slowly with compact steps",
        "A person edges forward with small careful movements",
        "A person takes short hesitant steps forward",
        "A person slowly inches forward one small step at a time",
        "A person walks forward with short precise steps",
        "A person moves forward slowly with small deliberate strides",
        "A person takes a few small steps forward and pauses",
        "A person shuffles forward gently with minimal foot lift",
        "A person creeps forward cautiously with tiny steps",
        "A person takes small slow steps forward while looking ahead",
        "A person moves forward with short careful footsteps",
        "A person takes small steps forward keeping close to the ground",
        "A person walks forward slowly with very controlled small steps",
        "A person inches ahead with small tentative movements",
        "A person takes compact forward steps at a gentle pace",
        "A person moves forward slowly with small even steps",
        "A person takes tiny forward steps while maintaining balance",
        "A person shuffles forward with short careful strides",
        "A person edges forward slowly with small steady steps",
        "A person takes three small steps forward and stops",
        "A person walks forward at a crawling pace with tiny steps",
        "A person moves ahead with small deliberate careful steps",
        "A person takes small forward steps keeping posture upright",
        "A person inches forward with controlled small movements",
        "A person walks forward with minimal stride length",
        "A person takes short gentle steps forward and pauses",
        "A person slowly moves forward with careful small strides",
        "A person takes several short steps forward at a steady pace",
        "A person moves forward very slowly with compact footsteps",
        "A person takes small measured steps forward with arms relaxed",
        "A person walks forward cautiously with tiny even steps",
    ],
}

# ===== 5. bend_small (小幅弯腰) =====
CATEGORIES["bend_small"] = {
    "durations": [3.0, 5.0],
    "prompts": [
        "A person bends forward slightly at the waist",
        "A person makes a small bow with a gentle forward lean",
        "A person leans forward slightly from the hips",
        "A person does a slight forward bend and returns upright",
        "A person tilts their upper body forward a little",
        "A person bends down slightly as if looking at something below",
        "A person makes a small forward lean and straightens back up",
        "A person does a gentle partial bow",
        "A person bends at the waist slightly and holds the position",
        "A person leans their torso forward just a bit",
        "A person does a small polite bow from the waist",
        "A person bends forward slightly to look at the ground",
        "A person makes a gentle nod with a small forward lean",
        "A person slightly bows forward and returns to standing",
        "A person tips their upper body forward in a small lean",
        "A person does a subtle forward bend at the hips",
        "A person leans down slightly as if to see something low",
        "A person makes a brief small bow and stands up straight",
        "A person bends forward a bit with hands at sides",
        "A person does a slight forward lean keeping back straight",
        "A person bends forward gently from the waist then recovers",
        "A person makes a mild forward bend and pauses",
        "A person leans their upper body forward modestly",
        "A person does a gentle forward tilt and returns upright",
        "A person bends slightly at the waist looking down",
        "A person makes a small respectful bow",
        "A person tips forward slightly at the hips then stands tall",
        "A person does a compact forward lean with straight legs",
        "A person bends forward a small amount and straightens slowly",
        "A person leans forward slightly then returns to neutral",
        "A person makes a small controlled forward bend",
        "A person gently bends at the waist and comes back up",
        "A person does a brief slight lean forward",
        "A person tilts forward a bit from the hips and holds",
        "A person bends forward just slightly keeping knees straight",
    ],
}

# ===== 6. bend_large (大幅弯腰) =====
CATEGORIES["bend_large"] = {
    "durations": [4.0, 7.0],
    "prompts": [
        "A person bends forward deeply at the waist reaching toward the ground",
        "A person does a deep bow bending their upper body far forward",
        "A person bends down low from the waist with a full forward fold",
        "A person leans forward deeply as if touching their toes",
        "A person bends at the waist going all the way down",
        "A person does a full deep bow with their torso nearly parallel to the ground",
        "A person bends down deeply to look at something on the ground",
        "A person makes a very deep forward bend from the hips",
        "A person bends forward at the waist until their hands reach their knees",
        "A person does a dramatic deep bow",
        "A person leans forward deeply with a straight back",
        "A person bends way down at the waist and slowly comes back up",
        "A person does a full forward fold reaching toward the floor",
        "A person bends deeply forward and holds the position",
        "A person leans their whole upper body forward in a deep bend",
        "A person bends at the waist deeply then straightens back up slowly",
        "A person does a deep respectful bow from the waist",
        "A person bends forward until their torso is horizontal",
        "A person makes a very low bow bending deeply at the hips",
        "A person bends down far to reach something near the ground",
        "A person does a deep forward bend with arms hanging down",
        "A person bends at the waist deeply while keeping legs straight",
        "A person leans far forward in a deep bow position",
        "A person bends down deeply and pauses before standing up",
        "A person makes a full deep bow and returns to standing",
        "A person bends forward deeply reaching past their knees",
        "A person does a slow deep forward fold",
        "A person bends their upper body forward as far as they can",
        "A person leans deeply forward from the hips letting arms dangle",
        "A person does a deep controlled bend at the waist",
        "A person bends forward all the way down then comes up slowly",
        "A person makes a profound deep bow from the waist",
        "A person bends down deeply to pick something up from the floor",
        "A person does a deep forward lean reaching for the ground",
        "A person bends at the waist going very low and holds",
    ],
}

# ===== 7. arms_horizontal (手臂平举) =====
CATEGORIES["arms_horizontal"] = {
    "durations": [3.0, 6.0],
    "prompts": [
        "A person raises both arms straight out to the sides at shoulder height",
        "A person extends both arms horizontally to form a T shape",
        "A person lifts their arms out to the sides until they are level with their shoulders",
        "A person holds both arms straight out to the sides",
        "A person slowly raises their arms to shoulder level on both sides",
        "A person extends their right arm straight out to the side at shoulder height",
        "A person holds their left arm out horizontally to the side",
        "A person raises both arms sideways to a horizontal position and holds",
        "A person extends both arms out wide at shoulder height",
        "A person lifts both arms up to shoulder level keeping them straight",
        "A person slowly spreads their arms out to the sides horizontally",
        "A person raises one arm out to the side at shoulder height",
        "A person extends both arms forward at shoulder height",
        "A person holds their arms out in front at chest level",
        "A person raises both arms forward horizontally",
        "A person lifts their arms up to the sides and holds them level",
        "A person extends their right arm forward at shoulder height",
        "A person holds their left arm out in front horizontally",
        "A person raises both arms to shoulder level forming a wide T pose",
        "A person slowly lifts arms to the sides until horizontal",
        "A person extends both arms out sideways with palms down",
        "A person raises arms horizontally to the sides and holds steady",
        "A person lifts both arms to shoulder height keeping them straight out",
        "A person extends one arm to the side and the other forward at shoulder height",
        "A person holds both arms straight out horizontally while standing still",
        "A person raises arms out to the sides slowly and deliberately",
        "A person extends both arms forward at shoulder level with straight elbows",
        "A person holds their arms in a T position at shoulder height",
        "A person slowly lifts arms sideways to horizontal and pauses",
        "A person raises right arm to the side and left arm forward at shoulder height",
        "A person extends both arms wide at shoulder height and holds the position",
        "A person lifts their arms out in a horizontal cross position",
        "A person slowly raises both arms to be level with shoulders",
        "A person extends arms straight out to the sides with control",
        "A person holds both arms at shoulder height with palms facing down",
        "A person raises arms horizontally and then slowly lowers them",
        "A person lifts arms to shoulder level, holds, then returns them down",
        "A person extends both arms to the sides and slowly rotates palms",
        "A person raises both arms to shoulder height and stands in a T pose",
        "A person slowly raises their arms out to the sides into a horizontal position",
    ],
}

# ===== 8. grasp_pick (抓取/拾取物品) =====
CATEGORIES["grasp_pick"] = {
    "durations": [4.0, 7.0],
    "prompts": [
        "A person reaches down and picks up a small object from the ground",
        "A person bends down and grabs something off the floor with one hand",
        "A person reaches forward and grasps an object at waist height",
        "A person picks up something from the ground using both hands",
        "A person reaches down to grab a small item near their feet",
        "A person bends at the waist to pick up an object from the floor",
        "A person reaches forward with their right hand to grab something",
        "A person reaches out with their left hand to take an object",
        "A person grabs something from a low shelf with one hand",
        "A person picks up an object from the ground and stands back up",
        "A person reaches down carefully and picks up a fragile object",
        "A person bends forward to grab something with both hands",
        "A person reaches out to grasp an object at chest height",
        "A person picks up a small item from the floor using their right hand",
        "A person reaches down and carefully lifts an object from the ground",
        "A person extends their arm to grab something in front of them",
        "A person bends down to pick up an object and holds it up",
        "A person reaches to the side to grab an object from a low surface",
        "A person picks up something small from the floor and examines it",
        "A person reaches forward with both hands and grasps an object",
        "A person bends to grab a light object from the ground",
        "A person reaches down to their left to pick something up",
        "A person reaches down to their right to grab a small object",
        "A person picks up an item from the floor slowly and carefully",
        "A person extends their arm down to pick up something near the ground",
        "A person reaches forward and takes an object from a surface",
        "A person bends their knees slightly and picks up something from the floor",
        "A person reaches out and grabs an object at arm's length",
        "A person picks up something heavy from the ground with both hands",
        "A person reaches down and grabs an object then brings it to their chest",
        "A person bends forward to grasp a small item from the floor",
        "A person reaches out in front and grasps an object with their right hand",
        "A person carefully picks up a delicate object from the ground",
        "A person reaches to the right side and grabs something from a low position",
        "A person reaches to the left and picks up an object from the floor",
        "A person bends down with one hand reaching to grab something",
        "A person picks up a cup from a low surface in front of them",
        "A person reaches down and picks up a book from the floor",
        "A person grabs an object from the ground and lifts it to waist height",
        "A person reaches forward to grab an object and pulls it toward them",
        "A person bends and picks up an object then holds it with both hands",
        "A person reaches down slowly and picks up something fragile from the floor",
        "A person reaches out with both arms and grabs a box in front of them",
        "A person picks up a light object from the ground with their left hand",
        "A person reaches down and collects an item from the floor",
    ],
}

# ===================================================================
# SUPPLEMENTARY CATEGORIES — single duration, 15 prompts each
# ===================================================================

CATEGORIES["weight_shift"] = {
    "durations": [5.0],
    "prompts": [
        "A person shifts their weight from one foot to the other while standing",
        "A person slowly transfers weight from left foot to right foot",
        "A person rocks gently from side to side while standing",
        "A person shifts their weight back and forth between feet",
        "A person sways slightly left and right while standing still",
        "A person transfers their weight forward onto their toes and back",
        "A person rocks their weight forward and backward gently",
        "A person shifts to lean on their left leg then their right leg",
        "A person slowly moves their center of gravity from left to right",
        "A person stands and gently sways shifting weight between feet",
        "A person leans slightly to the left then shifts to the right",
        "A person rocks their body weight from heel to toe",
        "A person shifts weight onto their right foot and holds",
        "A person transfers weight to their left foot and pauses",
        "A person sways gently forward and backward while standing",
    ],
}

CATEGORIES["squat_crouch"] = {
    "durations": [5.0],
    "prompts": [
        "A person does a slow partial squat and stands back up",
        "A person bends their knees slightly in a small squat",
        "A person lowers into a half squat and holds the position",
        "A person slowly squats down partway and returns to standing",
        "A person does a shallow squat keeping their back straight",
        "A person bends their knees into a quarter squat",
        "A person slowly crouches down partway then stands up",
        "A person does a controlled squat going halfway down",
        "A person lowers their body slightly by bending their knees",
        "A person squats down slowly with a controlled descent",
        "A person does a gentle squat and rises back up slowly",
        "A person crouches down low and then slowly stands up",
        "A person lowers into a deep squat position",
        "A person squats down fully and holds the position briefly",
        "A person slowly lowers into a squat keeping balance",
    ],
}

CATEGORIES["reaching"] = {
    "durations": [5.0],
    "prompts": [
        "A person reaches forward with their right arm extended",
        "A person reaches forward with their left arm extended",
        "A person reaches out with both arms in front of them",
        "A person extends their right arm to reach something in front",
        "A person extends their left arm to reach something to the side",
        "A person reaches up with their right hand above their head",
        "A person reaches up with their left hand above their head",
        "A person reaches up with both hands above their head",
        "A person stretches their arm out to reach something to the right",
        "A person stretches their arm out to reach something to the left",
        "A person reaches forward and down with one hand",
        "A person extends both arms forward at chest level",
        "A person reaches out to the side with their right arm",
        "A person reaches out to the side with their left arm",
        "A person stretches their arm up and forward to reach something high",
    ],
}

CATEGORIES["wave_gesture"] = {
    "durations": [5.0],
    "prompts": [
        "A person waves hello with their right hand",
        "A person waves hello with their left hand",
        "A person raises their hand and waves gently",
        "A person waves goodbye with one hand",
        "A person makes a small waving gesture",
        "A person waves at someone in front of them",
        "A person waves their hand back and forth at shoulder height",
        "A person makes a friendly wave with their right hand raised",
        "A person lifts their hand and waves slowly",
        "A person waves with their arm fully extended",
        "A person beckons someone to come closer with a hand gesture",
        "A person makes a come here gesture with their hand",
        "A person motions with their hand to beckon someone",
        "A person gestures with their right hand while talking",
        "A person makes a pointing gesture forward",
    ],
}

CATEGORIES["walk_carry"] = {
    "durations": [6.0],
    "prompts": [
        "A person walks forward while carrying a box with both hands",
        "A person walks forward while holding an object in their right hand",
        "A person walks forward while holding something in their left hand",
        "A person walks slowly carrying a tray with both hands",
        "A person walks forward holding a small object against their chest",
        "A person walks forward carrying something heavy with both arms",
        "A person walks forward while holding a cup in one hand",
        "A person walks carefully forward while carrying a fragile object",
        "A person walks forward with both arms holding a large box",
        "A person walks forward while holding an object out in front",
        "A person walks slowly while carrying something with two hands at waist height",
        "A person walks forward while holding a bag in their right hand",
        "A person walks forward while carrying a plate with both hands",
        "A person walks forward steadily while holding a box at chest level",
        "A person walks slowly while carefully carrying an object with both hands",
    ],
}

CATEGORIES["place_set_down"] = {
    "durations": [5.0],
    "prompts": [
        "A person sets a box down on the ground carefully",
        "A person places an object down on a surface in front of them",
        "A person slowly lowers a box to the ground with both hands",
        "A person puts down an object they were carrying",
        "A person bends down and sets something on the floor",
        "A person carefully places an item on the ground",
        "A person lowers an object from chest height to the ground",
        "A person sets down a tray on a low surface",
        "A person gently places something down in front of them",
        "A person bends forward and sets a package on the ground",
        "A person lowers an object they are holding to the floor",
        "A person carefully sets a fragile item down on the ground",
        "A person places a box on the ground and stands back up",
        "A person puts an object down on the floor with one hand",
        "A person sets something down gently to the left of them",
    ],
}

CATEGORIES["push_pull"] = {
    "durations": [5.0],
    "prompts": [
        "A person pushes something forward with both hands",
        "A person pushes a heavy object forward slowly",
        "A person pushes against something in front of them with both arms",
        "A person pushes forward with their arms extended",
        "A person pushes an object away from their body",
        "A person pulls something toward them with both hands",
        "A person pulls an object closer using both arms",
        "A person pulls something heavy toward them slowly",
        "A person reaches forward and pulls an object back",
        "A person pulls a handle toward their chest",
        "A person pushes something to the side with one hand",
        "A person pushes forward with one arm extended",
        "A person pulls something from the right side toward center",
        "A person pulls something from the left side toward center",
        "A person pushes a box forward along the ground",
    ],
}

CATEGORIES["torso_twist"] = {
    "durations": [5.0],
    "prompts": [
        "A person rotates their upper body to the left while standing",
        "A person rotates their upper body to the right while standing",
        "A person twists their torso to look behind them to the left",
        "A person twists their torso to look behind them to the right",
        "A person slowly rotates their torso from left to right",
        "A person twists their upper body back and forth gently",
        "A person turns their shoulders to the left while hips face forward",
        "A person turns their shoulders to the right while hips face forward",
        "A person does a slow torso rotation to the left and back",
        "A person does a slow torso rotation to the right and back",
        "A person twists their upper body to the left with arms out",
        "A person twists their upper body to the right with arms out",
        "A person rotates their torso slowly in a gentle twist",
        "A person swings their upper body gently from side to side",
        "A person does a controlled torso rotation looking over each shoulder",
    ],
}

CATEGORIES["stretch"] = {
    "durations": [5.0],
    "prompts": [
        "A person stretches both arms above their head",
        "A person does a full body stretch reaching upward",
        "A person stretches their arms up high and extends their back",
        "A person stretches their right arm across their chest",
        "A person stretches their left arm across their chest",
        "A person reaches up high with both arms in a morning stretch",
        "A person stretches by reaching both arms behind their back",
        "A person stretches their neck by tilting their head to the side",
        "A person stretches by extending both arms wide to the sides",
        "A person does a gentle back stretch leaning backward slightly",
        "A person stretches their upper body by reaching up and to the side",
        "A person stretches their arms up then slowly brings them down",
        "A person does a standing stretch arching their back slightly",
        "A person stretches their right arm above their head to the left",
        "A person stretches their left arm above their head to the right",
    ],
}

CATEGORIES["backward_walk"] = {
    "durations": [6.0],
    "prompts": [
        "A person walks backward slowly with a steady pace",
        "A person walks backward carefully step by step",
        "A person walks backward at a moderate pace",
        "A person steps backward steadily looking forward",
        "A person walks backward with controlled even steps",
        "A person backs up slowly while maintaining balance",
        "A person walks in reverse at a slow pace",
        "A person walks backward with careful footing",
        "A person moves backward at a walking pace",
        "A person walks backward slowly then stops",
        "A person walks backward and turns slightly to the left",
        "A person walks backward and turns slightly to the right",
        "A person walks backward in a gentle curve",
        "A person walks backward while looking over their shoulder",
        "A person retreats backward with steady steps",
    ],
}

CATEGORIES["diagonal_step"] = {
    "durations": [5.0],
    "prompts": [
        "A person walks diagonally forward to the left",
        "A person walks diagonally forward to the right",
        "A person takes steps diagonally forward and to the left",
        "A person takes steps diagonally forward and to the right",
        "A person moves at an angle walking forward and leftward",
        "A person moves at an angle walking forward and rightward",
        "A person walks forward on a diagonal path to the left slowly",
        "A person walks forward on a diagonal path to the right slowly",
        "A person steps diagonally backward to the left",
        "A person steps diagonally backward to the right",
        "A person walks at an angle going forward and sideways to the left",
        "A person walks at an angle going forward and sideways to the right",
        "A person takes a diagonal step forward and left then stops",
        "A person takes a diagonal step forward and right then stops",
        "A person walks diagonally forward at a slow controlled pace",
    ],
}

CATEGORIES["bow_greet"] = {
    "durations": [5.0],
    "prompts": [
        "A person makes a polite bow from the waist",
        "A person does a formal bow bending at the waist",
        "A person bows in greeting bending forward at the waist",
        "A person makes a respectful bow and returns to standing",
        "A person does a gentle bow of the head and upper body",
        "A person bows slightly in acknowledgment",
        "A person makes a deep formal bow",
        "A person bows slowly and returns to an upright position",
        "A person makes a quick small bow in greeting",
        "A person bows politely from the waist with arms at sides",
        "A person nods their head deeply in a greeting gesture",
        "A person makes a bow with their upper body going forward",
        "A person does a slow graceful bow",
        "A person bows forward then slowly rises back up",
        "A person makes a courteous bow bending about thirty degrees",
    ],
}

CATEGORIES["arm_lift_lower"] = {
    "durations": [5.0],
    "prompts": [
        "A person slowly raises both arms from their sides to above their head",
        "A person lifts their right arm from resting to shoulder height",
        "A person lifts their left arm from resting to shoulder height",
        "A person raises both arms forward from hip level to shoulder level",
        "A person slowly lowers both arms from shoulder height to their sides",
        "A person raises one arm up and then slowly lowers it back down",
        "A person lifts both arms up in front of them and lowers them slowly",
        "A person raises both arms overhead then brings them back down",
        "A person lifts their right arm up high and then lowers it",
        "A person lifts their left arm up high and then lowers it",
        "A person slowly raises arms to the sides then lowers them",
        "A person lifts arms forward to chest height then back down",
        "A person raises both arms sideways to shoulder level then lowers",
        "A person slowly raises right arm to the side and back down",
        "A person slowly raises left arm to the side and back down",
    ],
}

CATEGORIES["walk_gesture"] = {
    "durations": [6.0],
    "prompts": [
        "A person walks forward while waving their right hand",
        "A person walks forward while gesturing with one hand",
        "A person walks forward while pointing ahead with their right arm",
        "A person walks forward while waving with their left hand",
        "A person walks slowly while making a beckoning gesture",
        "A person walks forward and waves at someone to the side",
        "A person walks forward while raising one hand in greeting",
        "A person walks forward while motioning with their right hand",
        "A person walks slowly while gesturing to the left side",
        "A person walks slowly while gesturing to the right side",
        "A person walks forward while extending their arm to point at something",
        "A person walks and waves both hands above their head",
        "A person walks forward while giving a thumbs up",
        "A person walks forward and raises their hand to signal stop",
        "A person walks forward while reaching out with their right arm",
    ],
}

CATEGORIES["carry_heavy"] = {
    "durations": [6.0],
    "prompts": [
        "A person lifts a heavy box from the ground using both hands",
        "A person carries a heavy object with both arms close to their body",
        "A person slowly lifts a heavy object from the floor",
        "A person carries something heavy while walking carefully forward",
        "A person picks up a heavy box and carries it forward slowly",
        "A person struggles slightly while lifting a heavy object",
        "A person carries a large heavy box at waist height",
        "A person holds a heavy object against their chest while standing",
        "A person lifts a heavy item from the ground to waist level",
        "A person walks slowly while carrying a heavy object in both hands",
        "A person lifts a heavy box from the ground keeping their back straight",
        "A person holds a heavy object at chest height with both arms",
        "A person carries a large box forward with both hands and careful steps",
        "A person picks up something heavy with both hands and stands up",
        "A person slowly carries a heavy load forward step by step",
    ],
}

CATEGORIES["walk_backward_small"] = {
    "durations": [5.0],
    "prompts": [
        "A person walks forward then takes a few small steps backward",
        "A person walks forward briefly then backs up slowly",
        "A person takes a step forward then retreats with small backward steps",
        "A person walks forward stops then shuffles backward slightly",
        "A person walks then reverses with small careful backward steps",
        "A person moves forward then backs up a couple steps",
        "A person takes a few forward steps then gently moves backward",
        "A person walks forward stops and takes two small steps back",
        "A person walks forward then slowly reverses direction backward",
        "A person walks a few steps then carefully backs up",
        "A person walks forward pauses then takes small steps in reverse",
        "A person walks forward briefly then retreats slowly",
        "A person steps forward then immediately backs up with small steps",
        "A person walks forward then takes three small backward steps",
        "A person walks forward and then shuffles back slowly",
    ],
}

CATEGORIES["lean_tilt"] = {
    "durations": [5.0],
    "prompts": [
        "A person leans forward slightly from the ankles",
        "A person leans backward slightly with their whole body",
        "A person tilts forward gently keeping their body straight",
        "A person tilts backward a bit keeping their body rigid",
        "A person leans to the left side slightly",
        "A person leans to the right side slightly",
        "A person leans forward then returns to neutral",
        "A person leans backward then comes back upright",
        "A person tilts their body to the left then to the right",
        "A person sways forward slowly then sways back",
        "A person leans forward from the hips keeping legs straight",
        "A person leans back from the hips keeping legs straight",
        "A person tilts their upper body to the left and holds",
        "A person tilts their upper body to the right and holds",
        "A person leans forward slightly as if curious about something",
    ],
}

CATEGORIES["walk_slow_varied"] = {
    "durations": [7.0],
    "prompts": [
        "A person walks forward very slowly with an exaggerated careful gait",
        "A person walks forward slowly while looking down at the ground",
        "A person walks forward slowly with their hands behind their back",
        "A person walks forward cautiously as if on an uneven surface",
        "A person walks forward with a slow shuffling gait",
        "A person walks forward slowly with heavy tired steps",
        "A person walks forward slowly keeping their body low",
        "An old person walks forward slowly with careful steps",
        "A person walks forward at a very slow measured pace",
        "A person walks forward slowly dragging their feet slightly",
        "A person walks slowly forward with their head down",
        "A person walks forward with slow thoughtful steps",
        "A person walks forward slowly with arms swinging minimally",
        "A person walks forward very carefully on an imaginary narrow path",
        "A person walks forward slowly with their weight slightly forward",
    ],
}

CATEGORIES["stand_to_walk"] = {
    "durations": [6.0],
    "prompts": [
        "A person stands still then begins walking forward slowly",
        "A person starts from a standing position and begins to walk forward",
        "A person stands idle then takes their first steps forward",
        "A person transitions from standing still to walking forward",
        "A person begins walking from a stationary position",
        "A person starts standing and then slowly starts to walk",
        "A person stands for a moment then starts walking ahead",
        "A person initiates a walk from a standing rest position",
        "A person goes from standing still to taking slow steps forward",
        "A person stands then slowly begins moving forward",
        "A person starts to walk after standing still for a moment",
        "A person goes from a relaxed standing pose to walking forward",
        "A person begins their walk from a standstill position",
        "A person stands still then takes the first step forward slowly",
        "A person transitions from rest to a slow forward walk",
    ],
}

CATEGORIES["step_over"] = {
    "durations": [5.0],
    "prompts": [
        "A person steps over a low obstacle on the ground",
        "A person lifts one foot high to step over something",
        "A person carefully steps over an object in their path",
        "A person walks forward and steps over a small obstacle",
        "A person lifts their right leg high to step over something",
        "A person lifts their left leg high to step over something",
        "A person steps over a low barrier with one leg then the other",
        "A person carefully lifts each foot to step over an object",
        "A person walks and lifts their foot high over an obstacle",
        "A person steps over something low on the ground cautiously",
        "A person raises their knee high and steps over a barrier",
        "A person steps over an obstacle while walking forward slowly",
        "A person lifts their leg and steps carefully over something",
        "A person steps over a low object with exaggerated leg lift",
        "A person carefully walks over an obstacle raising each foot high",
    ],
}

CATEGORIES["walk_curve"] = {
    "durations": [7.0],
    "prompts": [
        "A person walks forward while gradually turning left",
        "A person walks forward while gradually turning right",
        "A person walks in a gentle curve to the left",
        "A person walks in a gentle curve to the right",
        "A person walks forward on a curving path to the left",
        "A person walks forward on a curving path to the right",
        "A person walks in a slow arc turning leftward",
        "A person walks in a slow arc turning rightward",
        "A person walks forward and curves gently to the left side",
        "A person walks forward and curves gently to the right side",
        "A person walks in a wide left turn at a slow pace",
        "A person walks in a wide right turn at a slow pace",
        "A person walks forward while slowly veering to the left",
        "A person walks forward while slowly veering to the right",
        "A person walks in a semicircle to the left",
    ],
}

CATEGORIES["weight_balance"] = {
    "durations": [5.0],
    "prompts": [
        "A person stands on one foot trying to keep balance",
        "A person balances on their right foot with left foot lifted slightly",
        "A person balances on their left foot with right foot lifted slightly",
        "A person shifts to stand on one leg and holds the position",
        "A person lifts one foot off the ground and balances",
        "A person stands on their right foot keeping balance with arms out",
        "A person stands on their left foot keeping balance with arms out",
        "A person tries to balance on one foot wobbling slightly",
        "A person stands on tiptoes and tries to keep balance",
        "A person balances on one foot then switches to the other",
        "A person lifts their left knee up and balances on right foot",
        "A person lifts their right knee up and balances on left foot",
        "A person stands on one foot with arms extended for balance",
        "A person balances carefully on their right leg",
        "A person balances carefully on their left leg",
    ],
}

CATEGORIES["approach_object"] = {
    "durations": [6.0],
    "prompts": [
        "A person walks forward toward something and stops close to it",
        "A person approaches an object in front of them slowly",
        "A person walks up to something and reaches out toward it",
        "A person walks forward and stops at arm's reach from an object",
        "A person slowly approaches something in front of them",
        "A person walks toward an object and leans forward to examine it",
        "A person walks forward and stops near something on the ground",
        "A person approaches something ahead and bends to look at it",
        "A person walks up to an object and reaches out with one hand",
        "A person slowly walks forward to get closer to something",
        "A person approaches an object and extends their arm to touch it",
        "A person walks forward then stops and examines something ahead",
        "A person moves toward an item and reaches down to it",
        "A person walks slowly toward something and pauses in front of it",
        "A person approaches and stands in front of an object",
    ],
}

CATEGORIES["shoulder_movement"] = {
    "durations": [5.0],
    "prompts": [
        "A person shrugs their shoulders up and down",
        "A person rolls their shoulders forward slowly",
        "A person rolls their shoulders backward slowly",
        "A person raises both shoulders up toward their ears",
        "A person lifts one shoulder and then the other alternately",
        "A person does a slow shoulder roll forward and backward",
        "A person shrugs their shoulders and lets them drop",
        "A person raises their right shoulder and lowers it",
        "A person raises their left shoulder and lowers it",
        "A person circles their shoulders forward several times",
        "A person circles their shoulders backward several times",
        "A person shrugs both shoulders up holds then drops them",
        "A person alternates shrugging left and right shoulders",
        "A person does a gentle shoulder shimmy from side to side",
        "A person rotates their shoulders in small circles",
    ],
}


# ---------------------------------------------------------------------------
# Script logic
# ---------------------------------------------------------------------------

def count_total_jobs(categories):
    """Count total prompt x duration pairs."""
    return sum(
        len(c["prompts"]) * len(c["durations"])
        for c in categories.values()
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generate motions for SONIC fine-tuning"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_finetune",
        help="Root output directory (default: outputs_finetune)",
    )
    parser.add_argument(
        "--model",
        default="Kimodo-SOMA-RP-v1.1",
        help="Kimodo model name (default: Kimodo-SOMA-RP-v1.1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of variations per prompt per duration (default: 3)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=100,
        help="Number of diffusion steps (default: 100)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only generate categories matching this substring",
    )
    parser.add_argument(
        "--bvh",
        action="store_true",
        help="Also export BVH (SOMA models only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and counts without generating",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cats = list(CATEGORIES.items())
    if args.category:
        cats = [(k, v) for k, v in cats if args.category in k]
        if not cats:
            print(f"No categories matching '{args.category}'")
            return

    cats_dict = OrderedDict(cats)
    total_jobs = count_total_jobs(cats_dict)
    total_samples = total_jobs * args.num_samples

    print(f"Categories: {len(cats_dict)}")
    print(f"Total jobs (prompt x duration): {total_jobs}")
    print(f"Samples per job: {args.num_samples}")
    print(f"Total data points: {total_samples}")
    print()

    if args.dry_run:
        for cat_name, cat_data in cats_dict.items():
            n = len(cat_data["prompts"])
            d = len(cat_data["durations"])
            dur_str = ",".join(f"{x:.0f}s" for x in cat_data["durations"])
            total = n * d * args.num_samples
            print(f"  {cat_name}: {n} prompts x {d} dur [{dur_str}] x {args.num_samples} = {total}")
        print(f"\nTotal: {total_jobs} jobs -> {total_samples} data points")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from kimodo import load_model
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import seed_everything

    model, resolved_name = load_model(
        args.model,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    print(f"Model: {resolved_name}")
    print(f"FPS: {model.fps}")
    print()

    generated = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for cat_name, cat_data in cats_dict.items():
        cat_dir = os.path.join(args.output_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        prompts = cat_data["prompts"]
        durations = cat_data["durations"]
        dur_str = ",".join(f"{x:.0f}s" for x in durations)

        print(f"=== {cat_name} ({len(prompts)} prompts x {len(durations)} dur [{dur_str}]) ===")

        for idx, prompt_text in enumerate(prompts):
            for duration in durations:
                dur_tag = f"d{duration:.0f}"
                name = f"{cat_name}_{idx:03d}_{dur_tag}"
                num_frames = int(duration * model.fps)

                expected = [
                    os.path.join(cat_dir, f"{name}_{i:02d}.npz")
                    for i in range(args.num_samples)
                ]

                if all(os.path.exists(f) for f in expected):
                    skipped += 1
                    continue

                seed = args.base_seed + hash(name) % 100000
                seed_everything(seed)

                try:
                    output = model(
                        [prompt_text],
                        [num_frames],
                        num_samples=args.num_samples,
                        num_denoising_steps=args.diffusion_steps,
                        multi_prompt=True,
                        post_processing=True,
                        return_numpy=True,
                    )

                    n = int(output["posed_joints"].shape[0])
                    for i in range(n):
                        single = {
                            k: (
                                v[i]
                                if hasattr(v, "shape")
                                and len(v.shape) > 0
                                and v.shape[0] == n
                                else v
                            )
                            for k, v in output.items()
                        }
                        save_kimodo_npz(expected[i], single)

                    if args.bvh and "somaskel" in model.skeleton.name:
                        from kimodo.exports.bvh import save_motion_bvh
                        from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

                        skeleton = model.skeleton
                        if isinstance(skeleton, SOMASkeleton30):
                            skeleton = skeleton.somaskel77.to(device)

                        for i in range(n):
                            bvh_path = expected[i].replace(".npz", ".bvh")
                            joints_pos = torch.from_numpy(output["posed_joints"][i]).to(device)
                            joints_rot = torch.from_numpy(output["global_rot_mats"][i]).to(device)
                            local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
                            root_positions = joints_pos[:, skeleton.root_idx, :]
                            save_motion_bvh(
                                bvh_path,
                                local_rot_mats,
                                root_positions,
                                skeleton=skeleton,
                                fps=model.fps,
                            )

                    generated += 1
                    elapsed = time.time() - t0
                    done = generated + skipped
                    rate = elapsed / max(generated, 1)
                    eta = rate * (total_jobs - done)
                    print(
                        f"  [{done}/{total_jobs}] {name} | "
                        f"{n} samples | "
                        f"{elapsed:.0f}s elapsed | "
                        f"ETA {eta:.0f}s"
                    )

                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {name}: {e}")
                    import traceback
                    traceback.print_exc()

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
    print(f"Generated: {generated} jobs ({generated * args.num_samples} samples)")
    print(f"Skipped:   {skipped} (already exist)")
    print(f"Failed:    {failed}")
    print(f"Output:    {args.output_dir}")


if __name__ == "__main__":
    main()
