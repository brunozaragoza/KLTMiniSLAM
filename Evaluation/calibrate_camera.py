import cv2
import numpy as np
import os

def calibrate_camera(video_path, pattern_size, skip_n):
    if not os.path.exists('calib_frames'):
        os.makedirs('calib_frames')

    video = cv2.VideoCapture(video_path)
    frame_number, saved_frame_number = 0, 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_number % skip_n == 0:
            frame_path = os.path.join('calib_frames', f'{saved_frame_number}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_frame_number += 1

        frame_number += 1

    video.release()
    print(f'Procesados {frame_number} frames, guardados {saved_frame_number} frames')

    objpoints, imgpoints = [], []
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for i in range(saved_frame_number):
        frame_path = os.path.join('calib_frames', f'{i}.jpg')
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, pattern_size, corners, ret)
        
        # Mostrar cada frame con el patrón detectado
        cv2.imshow('Frame', frame)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(objpoints) == 0:
        print("No se detectó ningún patrón en los frames guardados.")
        return None, None

    print(f'Se detectaron esquinas en {len(objpoints)} imágenes.')

    # Calibración
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savetxt('camera_matrix.txt', mtx)
    np.savetxt('distortion_coefficients.txt', dist)
    
    return mtx, dist

# Parámetros
video_path = 'calib2.mp4'
pattern_size = (9,6)  # Ajusta según el tablero
skip_n = 10  

camera_matrix, distortion_coefficients = calibrate_camera(video_path, pattern_size, skip_n)

#save the camera matrix and distortion coefficients in a file
np.savetxt('camera_matrix.txt', camera_matrix)
np.savetxt('distortion_coefficients.txt', distortion_coefficients)


if camera_matrix is not None:
    print('Matriz de la cámara:', camera_matrix)
    print('Coeficientes de distorsión:', distortion_coefficients)
