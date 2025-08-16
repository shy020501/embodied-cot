# import enum


# class CotTag(enum.Enum):
#     TASK = "TASK:"
#     PLAN = "PLAN:"
#     VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
#     SUBTASK_REASONING = "SUBTASK REASONING:"
#     SUBTASK = "SUBTASK:"
#     MOVE_REASONING = "MOVE REASONING:"
#     MOVE = "MOVE:"
#     GRIPPER_POSITION = "GRIPPER POSITION:"
#     ACTION = "ACTION:"


# def abbreviate_tag(tag: str):
#     return tag[0] + tag[-2]


# def get_cot_tags_list():
#     return [
#         CotTag.TASK.value,
#         CotTag.PLAN.value,
#         CotTag.VISIBLE_OBJECTS.value,
#         CotTag.SUBTASK_REASONING.value,
#         CotTag.SUBTASK.value,
#         CotTag.MOVE_REASONING.value,
#         CotTag.MOVE.value,
#         CotTag.GRIPPER_POSITION.value,
#         CotTag.ACTION.value,
#     ]


# def get_cot_database_keys():
#     return {
#         CotTag.TASK.value: "task",
#         CotTag.PLAN.value: "plan",
#         CotTag.VISIBLE_OBJECTS.value: "bboxes",
#         CotTag.SUBTASK_REASONING.value: "subtask_reason",
#         CotTag.SUBTASK.value: "subtask",
#         CotTag.MOVE_REASONING.value: "move_reason",
#         CotTag.MOVE.value: "move",
#         CotTag.GRIPPER_POSITION.value: "gripper",
#         CotTag.ACTION.value: "action",
#     }


import enum

class CotTag(enum.Enum):
#    TASK = "TASK:"
    # PLAN = "PLAN:"
#    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    # SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    # MOVE_REASONING = "MOVE REASONING:"
    # MOVE = "MOVE:"
#    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"  # optional, can be blank

def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]

def get_cot_tags_list():
    # Bridge 포맷에 최대한 맞춰 순서 정렬
    return [
#        CotTag.TASK.value,
        # CotTag.PLAN.value,
#        CotTag.VISIBLE_OBJECTS.value,
        # CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        # CotTag.MOVE_REASONING.value,
        # CotTag.MOVE.value,
#        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]

def get_cot_database_keys():
    return {
#        CotTag.TASK.value: "task",
        # CotTag.PLAN.value: "plan",
#        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        # CotTag.SUBTASK_REASONING.value: "subtask_reasoning",
        CotTag.SUBTASK.value: "subtask",
        # CotTag.MOVE_REASONING.value: "movement_reasoning",
        # CotTag.MOVE.value: "movement",
#        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",  # 실제로는 없음, "" 처리됨
    }
