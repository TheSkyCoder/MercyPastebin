# MercyMain (aka the main file)
# IMPORTANT: limitations. Currently the code works with fullscreen Overwatch in 16:9 mode on the primary monitor only!
# Another important note: this version of Mercy is kind of trashed, since it's quite a mess. Currently working on the new one...

# Also nice to note: the victory/defeat condition matchers is not currently 100% accurate, as the texts comes and go very fast and the OCR stuff tends to be slow.
# Currently we're running around 2fps; Future planned fix is to use some sort of machine learning / neural network algorithm to make everything run faster.

import cv2
import time
import numpy as np
import pytesseract
from mss import mss
from PIL import Image
from pprint import pprint
from fuzzywuzzy import fuzz
from imutils import contours
from fuzzywuzzy import process

# Set Tesseract dir here:
pytesseract.pytesseract.tesseract_cmd = "C:/Tesseract/tesseract.exe"

# SCT capture init
sct = mss()
displays = sct.enum_display_monitors()

# Fire bar detection color values. Remember, these are BGR!
firebar_blue_lower = np.array([0, 200, 70], dtype = "uint8")
firebar_blue_upper = np.array([255, 255, 150], dtype = "uint8")

# Kill cam detection threshold
killcam_detection_threshold = 65

# Seconds after first detection of killcam that ANOTHER killcam could then be registered
killcam_minimum_headway_length = 7

# Match result detection threshold
match_result_detection_threshold = 50

# Seconds after first detection of match result that ANOTHER match result could then be registered
match_result_minimum_headway_length = 70

# Killcam & match result condition headway parameters (internal; do not edit)
last_killcam_registered_on = 0
last_match_result_registered_on = 0

# Here you could "make stuff happen" when certain conditions are met:
def on_death():
    # Will be called upon killcam display (healer dead)
    print("Did you just died again?")

def on_victory():
    # Will be called upon end-of-game "victory" prompt
    print("Good job")

def on_defeat():
    # Will be called upon end-of-game "defeat" prompt
    print("Uh oh")

def on_tick():
    # Run once a tick (1 tick = 1 image processed)
    return

# Don't mess with this
def doNothing():
    return

# Hello world
print("Welcome to the automated Mercy performance evaluator!")
print("") #Too lazy to \n

# Main program loop
while 1:

    # Get screenfeed
    sct.get_pixels(displays[1])
    screenfeed_frame = Image.frombytes("RGB", (sct.width, sct.height), sct.image)

    # Transform screenfeed frame to BGR for OpenCV processing (for some reason COLOR_RGB2BGR dosen't seem to exist...)
    screenfeed_frame = cv2.cvtColor(cv2.cvtColor(np.array(screenfeed_frame), cv2.COLOR_RGB2HSV), cv2.COLOR_HSV2BGR)

    # Resize screenfeed to 720p for easier and lighter processing:
    process_frame = cv2.resize(np.array(screenfeed_frame), (1280, 720))
    grayscale_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY) # Only for display purposes

    # Grab frame(s) of interest:
    # note to self: StartY/EndY : StartX/EndX
    #player_data_frame = grayscale_frame[590:680,170:330]
    #ultimate_data_frame = grayscale_frame[570:650,600:700]
    #respawn_in_data_frame = grayscale_frame[0:100, 1000:1280]

    # Look out for killcams (that means our poor healer has died!)
    kill_cam_text_frame = process_frame[575:690,530:750]
    killcam_read = pytesseract.image_to_string(Image.fromarray(kill_cam_text_frame), config="-l eng+BigNoodle1")
    kill_cam_possibility = fuzz.ratio("KILL CAM", killcam_read)

    if kill_cam_possibility >= killcam_detection_threshold:
        # Kill cam detected. See if headway is OK?
        if int(time.time()) - last_killcam_registered_on >= killcam_minimum_headway_length:
            last_killcam_registered_on = int(time.time())
            print("[Info] Kill cam detected on " + str(last_killcam_registered_on))
            on_death()


    # Look out for victory/defeat texts:
    match_result_text_frame = process_frame[280:440,400:880]
    match_result_read = pytesseract.image_to_string(Image.fromarray(match_result_text_frame), config="-l eng+BigNoodle1")
    match_result_victory_possibility = fuzz.ratio("VICTORY!", match_result_read)
    match_result_defeat_possibility = fuzz.ratio("DEFEAT", match_result_read)

    # useful for debugging (sometimes) why the match result detector doesn't seem to work:
    #print("Wpos: " + str(match_result_victory_possibility) + ", Lpos: " + str(match_result_defeat_possibility))

    # To qualify for a valid GameResult, either victory possibility OR defeat possiblity MUST be greater than the threshold:
    if match_result_victory_possibility >= match_result_detection_threshold or match_result_defeat_possibility >= match_result_detection_threshold:
        # Check headway:
        if int(time.time()) - last_match_result_registered_on >= match_result_minimum_headway_length:
            last_match_result_registered_on = int(time.time())
            if match_result_victory_possibility > match_result_defeat_possibility:
                # Victory
                print("[Info] Match victory condition detected on " + str(last_match_result_registered_on))
                on_victory()
            else:
                # Defeat
                print("[Info] Match defeat condition detected on " + str(last_match_result_registered_on))
                on_defeat()

    # Try to evaluate player fire level; we absolutely need color on this!
    player_fire_frame = process_frame[645:680,170:330]

    # Mask & edge out the fire frame
    firebar_mask = cv2.inRange(player_fire_frame, firebar_blue_lower, firebar_blue_upper)
    firebar_edge = cv2.Canny(firebar_mask, 50, 150)
    firebar_edge = cv2.dilate(firebar_edge, None, iterations=2)
    firebar_edge = cv2.erode(firebar_edge, None, iterations=1)

    # TODO: process this and get length of fire-level line

    # Run the "on tick" function:
    on_tick()

    # We'll only need this for debugging:
    #cv2.imshow("Debug Frame", firebar_edge)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
