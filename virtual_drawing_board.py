import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector

class VirtualDrawingBoard:
    """
    A real-time virtual drawing board using computer vision to track hand movements.
    Use one hand to draw and select colors. Use one-handed gestures to dynamically 
    adjust the thickness of the pen or eraser.
    """
    
    def __init__(self, camera_index=0, width=1280, height=720):
        # Camera Settings
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
        
        # Hand Detector
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        
        # Canvas
        self.canvas = np.zeros((height, width, 3), np.uint8)
        
        # Drawing State
        self.xp, self.yp = 0, 0
        
        # Tools and Colors
        self.colors = {
            "BLUE": (255, 0, 0),
            "GREEN": (0, 255, 0),
            "RED": (0, 0, 255),
            "ERASER": (0, 0, 0)
        }
        self.current_tool = "BLUE"
        self.draw_color = self.colors[self.current_tool]
        
        # Thickness limits
        self.pen_thickness = 8
        self.eraser_thickness = 40
        
        self.min_thickness = 2
        self.max_pen_thickness = 50
        self.max_eraser_thickness = 150

    def draw_ui(self, img):
        """Draws the top header with color selection options and current active tool."""
        # Header background
        cv2.rectangle(img, (0, 0), (self.width, 100), (50, 50, 50), -1)
        
        # Header labels
        cv2.putText(img, "BLUE", (170, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["BLUE"], 3)
        cv2.putText(img, "GREEN", (360, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["GREEN"], 3)
        cv2.putText(img, "RED", (570, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["RED"], 3)
        cv2.putText(img, "ERASER", (760, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        # Draw selected tool banner at the bottom left
        status_text = f"Selected Tool: {self.current_tool}"
        if self.current_tool == "ERASER":
            status_text += f" (Size: {self.eraser_thickness})"
        else:
            status_text += f" (Size: {self.pen_thickness})"
            
        cv2.putText(img, status_text, (20, self.height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Visual indicator of current tool size
        preview_x, preview_y = 500, self.height - 40
        current_thickness = self.eraser_thickness if self.current_tool == "ERASER" else self.pen_thickness
        preview_color = (255, 255, 255) if self.current_tool == "ERASER" else self.draw_color
        cv2.circle(img, (preview_x, preview_y), current_thickness // 2, preview_color, -1)
        if self.current_tool == "ERASER":
            cv2.circle(img, (preview_x, preview_y), current_thickness // 2, (100, 100, 100), 2) # outline for eraser
            
    def handle_one_hand(self, img, hand):
        """Handles drawing, tool selection, and sizing logic using gestures."""
        lm_list = hand["lmList"]
        fingers = self.detector.fingersUp(hand)
        
        # Index finger tip
        x1, y1 = lm_list[8][0], lm_list[8][1]
        
        # Selection Mode: Index + Middle fingers ONLY
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
            self.xp, self.yp = 0, 0 # Reset drawing origin
            
            # Check if fingers are in the header region
            if y1 < 100:
                if 150 < x1 < 300:
                    self.current_tool = "BLUE"
                elif 350 < x1 < 500:
                    self.current_tool = "GREEN"
                elif 550 < x1 < 700:
                    self.current_tool = "RED"
                elif 750 < x1 < 900:
                    self.current_tool = "ERASER"
                
                self.draw_color = self.colors[self.current_tool]
                
        # Increase Size Mode: Thumb + Index fingers ONLY
        elif fingers[0] == 1 and fingers[1] == 1 and sum(fingers) == 2:
            self.xp, self.yp = 0, 0 # Stop drawing
            
            if self.current_tool == "ERASER":
                self.eraser_thickness = min(self.max_eraser_thickness, self.eraser_thickness + 1)
            else:
                self.pen_thickness = min(self.max_pen_thickness, self.pen_thickness + 1)
                
            cv2.putText(img, "INCREASING SIZE", (850, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Decrease Size Mode: Index + Pinky fingers ONLY
        elif fingers[4] == 1 and fingers[1] == 1 and sum(fingers) == 2:
            self.xp, self.yp = 0, 0 # Stop drawing
            
            if self.current_tool == "ERASER":
                self.eraser_thickness = max(self.min_thickness, self.eraser_thickness - 1)
            else:
                self.pen_thickness = max(self.min_thickness, self.pen_thickness - 1)
                
            cv2.putText(img, "DECREASING SIZE", (850, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Drawing / Erasing Mode: ONLY Index finger is up
        elif fingers[1] == 1 and sum(fingers) == 1:
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1
                
            current_thickness = self.eraser_thickness if self.current_tool == "ERASER" else self.pen_thickness
            
            # Draw line on canvas
            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, current_thickness)
            
            self.xp, self.yp = x1, y1
            
        else:
            # Standby mode (No drawing or scaling)
            self.xp, self.yp = 0, 0

    def handle_two_hands(self, img, hands):
        """Handles the Ruler (Scale) mode using two hands."""
        self.xp, self.yp = 0, 0 # Stop freehand drawing
        
        # Identify Left and Right hands
        if len(hands) == 2:
            if hands[0]["type"] == "Left":
                left_hand = hands[0]
                right_hand = hands[1]
            elif hands[1]["type"] == "Left":
                left_hand = hands[1]
                right_hand = hands[0]
            else:
                return
                
            left_fingers = self.detector.fingersUp(left_hand)
            right_fingers = self.detector.fingersUp(right_hand)
            
            # If left hand is completely open (4 or 5 fingers) -> Ruler Mode
            if sum(left_fingers) >= 4:
                cv2.putText(img, "RULER MODE ACTIVE", (850, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                
                # Points for the ruler (Index base and Pinky base of left hand)
                p1 = left_hand["lmList"][5][0:2]
                p2 = left_hand["lmList"][17][0:2]
                
                # Draw preview of the ruler on the screen
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 255, 255), 4)
                
                # If right hand has only index finger up, stamp the line onto the canvas!
                if right_fingers[1] == 1 and sum(right_fingers) == 1:
                    current_thickness = self.eraser_thickness if self.current_tool == "ERASER" else self.pen_thickness
                    cv2.line(self.canvas, (p1[0], p1[1]), (p2[0], p2[1]), self.draw_color, current_thickness)

    def run(self):
        """Main application loop."""
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        cv2.namedWindow("Virtual Smart Board", cv2.WINDOW_NORMAL)

        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Flip image horizontally for a mirror effect
            img = cv2.flip(img, 1)
            
            # Detect hands
            hands, img = self.detector.findHands(img, draw=True)
            
            # Merge canvas with original image. Canvas lines will be solid.
            img = cv2.addWeighted(img, 1, self.canvas, 1, 0)
            
            if hands:
                if len(hands) == 1:
                    self.handle_one_hand(img, hands[0])
                elif len(hands) == 2:
                    self.handle_two_hands(img, hands)
            else:
                self.xp, self.yp = 0, 0
                
            self.draw_ui(img)
            
            cv2.imshow("Virtual Smart Board", img)
            
            # Exit handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if cv2.getWindowProperty("Virtual Smart Board", cv2.WND_PROP_VISIBLE) < 1:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VirtualDrawingBoard()
    app.run()
