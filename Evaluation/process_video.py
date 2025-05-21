


#From a video file, extract frames and save them as images along with its timestamps in milliseconds
#The frames are saved in a folder named 'frames' in the current directory
#The timestamps are saved in a file named 'timestamps.txt' in the current directory

import cv2
import os


def process_video(video_path):
    # Create a folder to save the frames
    if not os.path.exists('Images'):
        os.makedirs('Images')

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Create a file to save the timestamps
    timestamps_file = open('timestamps.txt', 'w')

    # Extract frames and save them as images along with timestamps
    frame_number = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join('frames', f'{frame_number}.jpg')
        cv2.imwrite(frame_path, frame)

        # Get the timestamp of the frame
        timestamp = frame_number * 1000000 / fps
        timestamps_file.write(f'{frame_number} {timestamp}\n')

        frame_number += 1

    # Close the video file and the timestamps file
    video.release()
    timestamps_file.close()
    
    print(f'Processed {frame_number} frames')
    
    return frame_number

# Process the video file
video_path = 'video_2.mp4'
frame_count = process_video(video_path)
print(f'Processed {frame_count} frames')

