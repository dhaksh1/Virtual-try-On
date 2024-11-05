import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Path to shirt images
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)

# Constants for shirt sizing and display
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0

# Load button images
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)

# Counters and selection parameters
counterRight = 0
counterLeft = 0
selectionSpeed = 10
buttonDebounce = 30  # Delay in frames before allowing another button press
lastButtonPress = 0  # Frame count to track debounce

# Smoothing parameters
previousLm11 = np.array([0, 0])
previousLm12 = np.array([0, 0])
smoothingFactor = 0.1  # Controls the smoothing, adjust for more or less smoothness

# Create a full-screen window
cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frameCount = 0  # To keep track of frame counts for debounce

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Resize the camera feed to fit the screen while maintaining aspect ratio
    img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        # Get current landmarks for shoulder points (11 and 12)
        lm11 = np.array(lmList[11][0:2])  # Left shoulder
        lm12 = np.array(lmList[12][0:2])  # Right shoulder

        # Apply smoothing to avoid flickering
        lm11 = smoothingFactor * lm11 + (1 - smoothingFactor) * previousLm11
        lm12 = smoothingFactor * lm12 + (1 - smoothingFactor) * previousLm12

        previousLm11 = lm11  # Update for next frame
        previousLm12 = lm12  # Update for next frame

        # Calculate width of shirt and ensure it's a valid value
        shoulderWidth = int(lm11[0] - lm12[0])
        widthOfShirt = int(shoulderWidth * fixedRatio)

        if widthOfShirt > 0:  # Ensure width is positive before resizing
            # Load the current shirt image from the folder
            shirtPath = os.path.join(shirtFolderPath, listShirts[imageNumber])
            imgShirt = cv2.imread(shirtPath, cv2.IMREAD_UNCHANGED)

            if imgShirt is None:
                print(f"Error: Could not load shirt image {shirtPath}")
                continue

            # Resize shirt image based on shoulder width and maintain aspect ratio
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = shoulderWidth / 190

            # Adjust offset based on shoulder position
            offsetX = int(44 * currentScale)
            offsetY = int(60 * currentScale)  # Adjusting Y to ensure it fits the body

            try:
                # Apply the shirt image on the body using overlayPNG
                shirtPositionX = int(lm12[0] - offsetX)  # Position relative to right shoulder
                shirtPositionY = int(lm12[1] - offsetY)  # Position relative to shoulder height

                img = cvzone.overlayPNG(img, imgShirt, (shirtPositionX, shirtPositionY))
            except Exception as e:
                print(f"Error overlaying image: {e}")
        else:
            print(f"Invalid shirt width: {widthOfShirt}")

        # Overlay buttons (adjust the position if needed based on screen resolution)
        img = cvzone.overlayPNG(img, imgButtonRight, (1070, 360))  # Adjusted position
        img = cvzone.overlayPNG(img, imgButtonLeft, (70, 360))  # Adjusted position

        # Right button logic for changing shirt (with debounce)
        if lmList[16][0] < 300 and lmList[16][2] < 400 and frameCount - lastButtonPress > buttonDebounce:  # Checking y position
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1
                lastButtonPress = frameCount  # Update debounce time
        # Left button logic for changing shirt (with debounce)
        elif lmList[15][0] > 900 and lmList[15][2] < 400 and frameCount - lastButtonPress > buttonDebounce:  # Checking y position
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
                lastButtonPress = frameCount  # Update debounce time
        else:
            counterRight = 0
            counterLeft = 0

    # Display the full-screen image
    cv2.imshow("Image", img)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment the frame count
    frameCount += 1

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
