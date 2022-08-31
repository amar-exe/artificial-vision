#%% Section 3.1 - Regular Video
import cv2
cap_video = cv2.VideoCapture(0)
while(True):
    
    # Capture frame-by-frame
    ret, frame = cap_video.read()

    # Visualise the captured frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# And release the capture when everything is done
cap_video.release()
cv2.destroyAllWindows()

#%% Section 3.1 - Grayscale Video
import cv2
cap_video = cv2.VideoCapture(0)
while(True):
    
    # Capture frame-by-frame
    ret, frame = cap_video.read()

    # Convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the captured frame
    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# And release the capture when everything done
cap_video.release()
cv2.destroyAllWindows()

#%% Section 3.2 - Saving Videos
import cv2
import sys

path = ('/Users/Amar/Downloads/output.avi')

cap_video = cv2.VideoCapture(0)

if not cap_video:
    print('Failed VideoCapture: invalid parameter!')
    
cap_video.set(1, 20.0) # Match fps
cap_video.set(3, 1280) # Match width
cap_video.set(4, 720) # Match height
    
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_video = cv2.VideoWriter(path, fourcc, 20.0, (1280,720))
    
if not out_video:
    print('Failed VideoWritter: invalid parameters')
    sys.exit(1)
    
flip_flag = 0
flip_counter = 0
time_limit = 0

# Write the frame
while (cap_video.isOpened()):
    ret, frame = cap_video.read()
    if ret==True:
        
        # Flipping the frame (1)
        if flip_counter < 30:
                
            flip_flag = 1
            flip_counter += 1
                
        elif flip_counter < 60:
            flip_flag = 0
            flip_counter += 1
                
        else:
            flip_flag = 0
            flip_counter = 0
                
        frame = cv2.flip(frame, flip_flag)

        # Write the flipped frame
        out_video.write(frame)

        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        time_limit += 1
 
    else:
        break

    if time_limit == 1000: # (2)
        break
# Release everything
cap_video.release()
out_video.release()
cv2.destroyAllWindows()

#%% Section 3.3 - Playing Video from File
import cv2
cap_vid = cv2.VideoCapture('/Users/Amar/Downloads/Traffic_counting.mp4')
while(cap_vid.isOpened()):
    ret, frame = cap_vid.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap_vid.release()
cv2.destroyAllWindows()
