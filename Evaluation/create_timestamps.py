import os

def create_timestamps_file(directory, output_file):
    with open(output_file, 'w') as file:
        filenames = sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]))
        for filename in filenames:
            if filename.endswith('.jpg'):  
                timestamp = os.path.splitext(filename)[0]
                file.write(timestamp + '\n')


directory = '/home/user/datasets/Own_video_2/Images/'
output_file = 'own_vid_2.txt'  

create_timestamps_file(directory, output_file)