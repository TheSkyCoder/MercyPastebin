# The new MercyMain
# Now more accurate and less reliant on OCR. Still a work in progress though.
#
# By TheSkyCoder (/u/TheHSSkyCoder). Licensed under the MIT license.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
# AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import cv2
import time
import collections
import pytesseract
import numpy as np
from mss import mss
from PIL import Image
from fuzzywuzzy import fuzz

# Debug me?
debug_mode = True
debug_show_frame = False
debug_level = 1

# Set Tesseract dir here:
pytesseract.pytesseract.tesseract_cmd = "C:/Tesseract/tesseract.exe"

# Kill cam detection threshold
killcam_detection_threshold = 65

# Seconds after first detection of killcam that ANOTHER killcam could then be registered
killcam_minimum_headway_length = 7

# Seconds after first detection of match result that ANOTHER match result could then be registered
match_result_minimum_headway_length = 60

# Feature mapping thresholds
victory_condition_mapping_threshold = 3
kill_cam_feature_mapping_threshold = 1 # should not be modified

# Game result detection threshold
game_result_detection_threshold = 4000

# Fire bar detection color values. Remember, these are BGR!
firebar_blue_lower = np.array([0, 200, 70], dtype = "uint8")
firebar_blue_upper = np.array([255, 255, 150], dtype = "uint8")

# Victory condition color values. Remember, these are BGR!
victory_yellow_lower = np.array([83, 200, 200], dtype = "uint8")
victory_yellow_upper = np.array([91, 215, 255], dtype = "uint8")

# Defeat condition color values. Remember, these are BGR!
defeat_red_lower = np.array([30, 0, 200], dtype = "uint8")
defeat_red_upper = np.array([60, 10, 255], dtype = "uint8")

# ************************ DO NOT EDIT VARIABLES BELOW THIS LINE ************************ #

# SCT capture init
sct = mss()
displays = sct.enum_display_monitors()

# Killcam & match result condition headway parameters (internal; do not edit)
last_killcam_registered_on = 0
last_match_result_registered_on = 0

# Normalized firebar level (over 10 ticks)
normalized_firebar_level = 0

# Firebar average over time
firebar_average_ot = collections.deque([0, 0, 0, 0, 0, 0, 0])

# Feature mapping base images
kill_cam_feature_image = cv2.imread("killcam.jpg")

# Initiate feature detector
orb = cv2.ORB_create()

# Pre-calculate keypoints for the "Kill Cam" feature detector
killcam_text_keypoints = orb.detect(kill_cam_feature_image, None)
killcam_text_keypoints, killcam_text_descriptors = orb.compute(kill_cam_feature_image, killcam_text_keypoints)

# Matcher stuff
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ==================================== USER-EDITABLE FUNCTIONS STARTS HERE ====================================
def on_victory():
    # Called when a victory condition is detected
    print("\n*** Victory! ***\n")

def on_defeat():
    # Called when a defeat condition is detected
    print("\n*** Defeat! ***\n")

def on_death():
    # Called on player death
    print("\n*** Did you just died again? ***\n")

def on_tick(firebar_average = 0):
    # Run once a "tick" (i.e. one per frame processed)
    return
# ===================================== USER-EDITABLE FUNCTIONS ENDS HERE =====================================

# Function: look out for "victory" or "defeat" texts
# We still need OCR on this since the feature map seems to can't distinguish between "Victory" and "Defeat",
# But the feature guide will help us increase FPS by doing OCR only when necessary.
def findMatchResultsCondition( screen_frame_descriptors, screen_frame ):

    # Global var
    global last_match_result_registered_on

    # Our "Frame of interest"
    match_result_text_frame = screen_frame[280:440,400:880]

    # Mask & score calculation for the "defeat" text
    defeat_red_mask = cv2.inRange(match_result_text_frame, defeat_red_lower, defeat_red_upper)
    defeat_red_edge = cv2.Canny(defeat_red_mask, 50, 150)
    defeat_red_edge = cv2.dilate(defeat_red_edge, None, iterations=2)
    defeat_red_edge = cv2.erode(defeat_red_edge, None, iterations=1)
    defeat_score = cv2.countNonZero(defeat_red_edge)

    # Mask & score calculation for the "victory" text
    victory_yellow_mask = cv2.inRange(match_result_text_frame, victory_yellow_lower, victory_yellow_upper)
    victory_yellow_edge = cv2.Canny(victory_yellow_mask, 50, 150)
    victory_yellow_edge = cv2.dilate(victory_yellow_edge, None, iterations=2)
    victory_yellow_edge = cv2.erode(victory_yellow_edge, None, iterations=1)
    victory_score = cv2.countNonZero(victory_yellow_edge)

    if debug_mode and debug_level >= 2:
        print("[Debug] Victory score: " + str(victory_score) + ", Defeat score: " + str(defeat_score))

    # Score above threshold?
    if victory_score >= game_result_detection_threshold or defeat_score >= game_result_detection_threshold:
        # Check headway, should we continue?
        if int(time.time()) - last_match_result_registered_on >= match_result_minimum_headway_length:

            # Mark last result time
            last_match_result_registered_on = int(time.time())

            # Does it looks like a victory or a defeat?
            if victory_score >= defeat_score:
                if debug_mode:
                    print("[Debug] Victory condition, score: " + str(victory_score))
                on_victory()
            else:
                if debug_mode:
                    print("[Debug] Defeat condition, score: " + str(defeat_score))
                on_defeat()
        else:
            if debug_mode:
                print("[Debug] Victory/Defeat condition detected but not registered due to headway restrictions (last detected at " + str(last_match_result_registered_on) + ")")
                print("[Debug] Victory score: " + str(victory_score) + ", Defeat score: " + str(defeat_score))
        # We found something!
        return True
    else:
        return False

# Function to find kill cams (i.e. our healer has been killed...)
def findKillcam( screen_frame_descriptors, screen_frame ):

    # Again, global stuff:
    global last_killcam_registered_on

    # Feature matching for "Kill Cam" text
    killcam_text_matches = bfMatcher.match(screen_frame_descriptors, killcam_text_descriptors)
    killcam_text_dist = [m.distance for m in killcam_text_matches]
    killcam_text_divider = len(killcam_text_dist) if len(killcam_text_dist) > 0 else 1 # Prevent division by zero with empty arrays
    killcam_text_threshold_dist = (sum(killcam_text_dist) / killcam_text_divider) * 0.5
    killcam_text_matches = [m for m in killcam_text_matches if m.distance < killcam_text_threshold_dist]

    if len(killcam_text_matches) >= kill_cam_feature_mapping_threshold: # This threshold should be low, hence the "should not be edited" above

        if debug_mode:
            print("[Debug] Possible killcam condition found with a M-score of " + str(len(killcam_text_matches)))

        # Possible kill-cam detected. Launch OCR!
        # TODO: Use something that ISN'T OCR here, so the user would not need to run Tesseract
        kill_cam_text_frame = screen_frame[575:690,530:750]
        killcam_read = pytesseract.image_to_string(Image.fromarray(kill_cam_text_frame), config="-l eng+BigNoodle1")
        kill_cam_possibility = fuzz.ratio("KILL CAM", killcam_read)

        if debug_mode and debug_level >= 2:
            print("[Debug] Kill cam possibility: " + str(kill_cam_possibility))

        if debug_mode and debug_level >= 3:
            print("[Debug] Kill cam OCR Output: " + str(killcam_read))

        if kill_cam_possibility >= killcam_detection_threshold:
            # Kill cam detected. See if headway is OK?
            if int(time.time()) - last_killcam_registered_on >= killcam_minimum_headway_length:
                last_killcam_registered_on = int(time.time())
                if debug_mode:
                    print("[Debug] Kill cam detected on < " + str(last_killcam_registered_on) + " >")
                on_death()
            else:
                if debug_mode and debug_level >= 2:
                    print("[Debug] Kill cam detected (but not registered due to headway restrictions) on " + str(time.time()))

# Function to process firebar:
def readFirebar( screen_frame ):

    # Global stuff yet again
    global firebar_average_ot
    global normalized_firebar_level

    # Try to evaluate player fire level
    player_fire_frame = screen_frame[645:680,170:330]

    # Mask & edge out the fire frame
    firebar_mask = cv2.inRange(player_fire_frame, firebar_blue_lower, firebar_blue_upper)
    firebar_edge = cv2.Canny(firebar_mask, 50, 150)
    firebar_edge = cv2.dilate(firebar_edge, None, iterations=2)
    firebar_edge = cv2.erode(firebar_edge, None, iterations=1)
    firebar_score = cv2.countNonZero(firebar_edge)

    if debug_mode and debug_show_frame:
        cv2.imshow("Firebar", firebar_edge)

    # Push to normalization list:
    if firebar_score != 0:
        firebar_average_ot.appendleft(firebar_score)
        firebar_average_ot.pop()

    if debug_mode and debug_level >= 3:
        print("[Debug] Firebar score: " + str(firebar_score))
        #print(firebar_average_ot)

    # Calculate "normalized" firebar level
    normalized_firebar_score = sum(firebar_average_ot) / len(firebar_average_ot)

    if debug_mode and debug_level >= 2:
        print("[Debug] Normalized firebar score: " + str(normalized_firebar_score))
        #print(firebar_average_ot)

    return normalized_firebar_score

# Tell the user if we're running in debug?
if debug_mode:
    print("Running in DEBUG mode, debug level: " + str(debug_level))

# Main program loop
while 1:

    if debug_mode and debug_level >= 2:
        print("========== DEBUG MODE: " + "%0.3f" % round(time.time(), 3) + " ==========")

    # Get screenfeed
    sct.get_pixels(displays[1])
    screenfeed_frame = Image.frombytes("RGB", (sct.width, sct.height), sct.image)

    # Transform screenfeed frame to BGR for OpenCV processing (for some reason COLOR_RGB2BGR dosen't seem to exist...)
    screenfeed_frame = cv2.cvtColor(cv2.cvtColor(np.array(screenfeed_frame), cv2.COLOR_RGB2HSV), cv2.COLOR_HSV2BGR)

    # Resize screenfeed to 720p for easier and lighter processing:
    process_frame = cv2.resize(np.array(screenfeed_frame), (1280, 720))

    # Find keypoints
    screenfeed_keypoints = orb.detect(process_frame, None)
    screenfeed_keypoints, screenfeed_descriptors = orb.compute(process_frame, screenfeed_keypoints)

    # Look for match results and killcams
    findMatchResultsCondition(screenfeed_descriptors, process_frame)
    findKillcam(screenfeed_descriptors, process_frame)

    # Call the on-tick function
    on_tick(firebar_average = readFirebar(process_frame))

    if debug_mode and debug_level >= 2:
        # For readability with long debugging texts
        print("\n")

    if debug_mode and debug_show_frame:
        keypoints_debug_frame = cv2.drawKeypoints(process_frame, screenfeed_keypoints, None, color=(0,255,0), flags=0)
        cv2.imshow("Debug Frame", keypoints_debug_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
