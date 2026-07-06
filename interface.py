import tkinter 
import cv2 
import pytorch

    # ---------------- UI ----------------
    cv2.putText(frame, f"CARS: {car}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"BIKES: {bike}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"PEOPLE: {person}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"SPEED LIMIT: {SPEED_LIMIT}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("AI TRAFFIC SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
