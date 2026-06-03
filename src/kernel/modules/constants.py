"""
Constants for Atlético Intelligence offside detection system.
"""

# Keypoint indices — COCO 17-keypoint layout (used by YOLO11x-pose)
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle
FOOT_KEYPOINTS = [15, 16]  # left_ankle (15), right_ankle (16)
HEAD_KEYPOINTS = [3, 4]    # left_ear (3), right_ear (4)

# Team colors (BGR format for OpenCV)
TEAM_COLORS = {
    'attacking': (0, 255, 0),      # Green
    'defending': (0, 0, 255),      # Red
    'ball': (0, 255, 255),         # Yellow
    'referee': (255, 0, 0),        # Blue
    'offside_line': (255, 255, 0), # Cyan
}

# Video processing
DEFAULT_TARGET_FPS = 30
DEFAULT_TARGET_RESOLUTION = (640, 480)
CLIP_DEFAULT_DURATION_SEC = None  # full video — no clipping

# Detection
MIN_PLAYERS_FOR_OFFSIDE = 2
DETECTION_CONFIDENCE_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.6

# K-means clustering
KMEANS_N_CLUSTERS = 2
KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 10

# Jersey color extraction (torso region)
TORSO_START_PERCENT = 0.3  # 30% from top
TORSO_END_PERCENT = 0.7    # 70% from top
TORSO_X_MARGIN_PERCENT = 0.2  # 20% margin from sides

# Ball detection
BALL_MIN_AREA = 20
BALL_MAX_AREA = 500
BALL_MIN_RADIUS = 3
BALL_MAX_RADIUS = 30
BALL_CIRCULARITY_THRESHOLD = 0.7

# Visualization (3D diagrams)
DIAGRAM_WIDTH = 800
DIAGRAM_HEIGHT = 600
DIAGRAM_DPI = 100
DIAGRAM_BGCOLOR = 'white'
DIAGRAM_FIELD_COLOR = 'green'
DIAGRAM_LINE_COLOR = 'white'
DIAGRAM_PLAYER_SIZE = 100
DIAGRAM_BALL_SIZE = 50
DIAGRAM_DEFENDER_COLOR = 'red'
DIAGRAM_ATTACKER_COLOR = 'blue'
DIAGRAM_OFFSIDE_LINE_COLOR = 'yellow'

# Logging
LOG_FORMAT_JSON = True
LOG_LEVEL = 'INFO'

# Incident types
INCIDENT_TYPE_OFFSIDE = 'offside'
INCIDENT_TYPE_GOAL = 'goal'

# Verdict constants
VERDICT_OFFSIDE = 'OFFSIDE'
VERDICT_ONSIDE = 'ONSIDE'
VERDICT_GOAL = 'GOAL'
VERDICT_NO_GOAL = 'NO_GOAL'
VERDICT_UNCERTAIN = 'UNCERTAIN'
