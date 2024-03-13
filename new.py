
import dlib
import cv2
from pydub import AudioSegment
import numpy as np
image_path = 'C:\\Users\\divya\\Desktop\\cartoonizer\\image.png'  # Replace with the actual image path
audio_path = 'C:\\Users\\divya\\Desktop\\cartoonizer\\harvard.wav'  # Replace with the actual audio file path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\divya\\Desktop\\cartoonizer\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat')  # Replace with the actual path
audio = AudioSegment.from_wav(audio_path)
audio_data = np.array(audio.get_array_of_samples())
faces = detector(gray)
shape = predictor(gray, faces[0])
landmarks = np.array([[p.x, p.y] for p in shape.parts()])
frame_duration = int(len(audio_data) / (len(landmarks) / 25))
video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (image.shape[1], image.shape[0]))
for i in range(0, len(audio_data), frame_duration):
    frame = image.copy()
    audio_segment = audio_data[i:i + frame_duration]

    # Calculate the mouth region
    mouth_top = np.min(landmarks[48:61, 1]) - 10
    mouth_bottom = np.max(landmarks[48:61, 1]) + 10
    mouth_left = np.min(landmarks[48:61, 0]) - 10
    mouth_right = np.max(landmarks[48:61, 0]) + 10

    # Draw the modified mouth region on the frame
    cv2.rectangle(frame, (mouth_left, mouth_top), (mouth_right, mouth_bottom), (0, 0, 255), cv2.FILLED)

    video_writer.write(frame)
video_writer.release()
print("Lip sync animation generated successfully!")
