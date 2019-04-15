import numpy as np
from pathlib import Path
import json
def readOp3d(path,pattern='*ts.json',mode=0):
    video = json_pack2(path, is3D=True, pattern=pattern,mode=mode)
    k=67
    if mode==0:
        k=67  #both hand + body
    elif mode==1:
        k=25 #body
    pose, label = video_info_parsing3D(video, keypoints=k)

    return  pose

def readOp2d(path,pattern='*ts.json',mode=0):
    video = json_pack2(path, is3D=False, pattern=pattern,mode=mode)
    k = 67
    if mode == 0:
        k = 67  # both hand + body
    elif mode == 1:
        k = 25  # body
    pose, label = video_info_parsing2D(video, keypoints=k)
    return pose

def video_info_parsing2D(video_info,keypoints=18, num_person_in=1, num_person_out=1):
    data_numpy = np.zeros((3, len(video_info['data']), keypoints, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::2]
            data_numpy[1, frame_index, :, m] = pose[1::2]
            data_numpy[2, frame_index, :, m] = score

    # centralization
    data_numpy[0:2] = data_numpy[0:2] - 0.5
    data_numpy[0][data_numpy[2] == 0] = 0
    data_numpy[1][data_numpy[2] == 0] = 0

    sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                   0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    label = video_info['label_index']
    return data_numpy, label


def video_info_parsing3D(video_info, keypoints=18, num_person_in=1, num_person_out=1):
    dim=4
    data_numpy = np.zeros((dim, len(video_info['data']), keypoints, num_person_in))
    for frame_info in video_info['data']:
        frame_index = frame_info['frame_index']
        for m, skeleton_info in enumerate(frame_info["skeleton"]):
            if m >= num_person_in:
                break
            pose = skeleton_info['pose']
            score = skeleton_info['score']
            data_numpy[0, frame_index, :, m] = pose[0::3]
            data_numpy[1, frame_index, :, m] = pose[1::3]
            data_numpy[2, frame_index, :, m] = pose[2::3]
            data_numpy[3, frame_index, :, m] = score

    # centralization

   # data_numpy[0:3] = data_numpy[0:3] - 0.5
    #data_numpy[0][data_numpy[3] == 0] = 0
    #data_numpy[1][data_numpy[3] == 0] = 0
    #data_numpy[2][data_numpy[3] == 0] = 0

    sort_index = (-data_numpy[3, :, :, :].sum(axis=1)).argsort(axis=1)
    for t, s in enumerate(sort_index):
        data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                   0))
    data_numpy = data_numpy[:, :, :, :num_person_out]

    label = video_info['label_index']
    return data_numpy, label

def json_pack2(snippets_dir, frame_width=1, frame_height=1,is3D=False , label='unknown', label_index=-1,pattern="*ts.json",mode=0):
    sequence_info = []
    p = Path(snippets_dir)
    id=0
    for path in p.glob(pattern):
        json_path = str(path)
        frame_id = id
        id+=1
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            score, coordinates = [], []
            skeleton = {}
            if is3D:
                keypoints = person['pose_keypoints_3d']
                if len(keypoints) != 100:
                    for i in range(25):
                        coordinates += [0, 0, 0]
                        score += [0]
                for i in range(0, len(keypoints), 4):
                    coordinates += [keypoints[i] , keypoints[i + 1] ,keypoints[i + 2]]
                    score += [keypoints[i + 3]]
                if mode==0:
                    keypoints = person['hand_left_keypoints_3d']
                    if len(keypoints) != 84:
                        for i in range(21):
                            coordinates += [0, 0, 0]
                            score += [0]
                    for i in range(0, len(keypoints), 4):
                        coordinates += [keypoints[i], keypoints[i + 1], keypoints[i + 2]]
                        score += [keypoints[i + 3]]
                    keypoints = person['hand_right_keypoints_3d']
                    if len(keypoints)!=84:
                            for i in range(21):
                                coordinates +=[0,0,0]
                                score += [0]
                    for i in range(0, len(keypoints), 4):
                        coordinates += [keypoints[i], keypoints[i + 1], keypoints[i + 2]]
                        score += [keypoints[i + 3]]
            else:
                keypoints = person['pose_keypoints_2d']
                if len(keypoints)!=75:
                        for i in range(25):
                            coordinates +=[0,0]
                            score += [0]
                for i in range(0, len(keypoints), 3):
                    coordinates += [keypoints[i]/frame_width, keypoints[i + 1]/frame_height]
                    score += [keypoints[i + 2]]
                if mode==0:
                    keypoints = person['hand_left_keypoints_2d']
                    if len(keypoints)!=63:
                            for i in range(21):
                                coordinates +=[0,0]
                                score += [0]
                    for i in range(0, len(keypoints), 3):
                        coordinates += [keypoints[i] / frame_width, keypoints[i + 1] / frame_height]
                        score += [keypoints[i + 2]]

                    keypoints = person['hand_right_keypoints_2d']
                    if len(keypoints)!=63:
                            for i in range(21):
                                coordinates +=[0,0]
                                score += [0]
                    for i in range(0, len(keypoints), 3):
                        coordinates += [keypoints[i] / frame_width, keypoints[i + 1] / frame_height]
                        score += [keypoints[i + 2]]


            skeleton['pose'] = coordinates
            skeleton['score'] = score
            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index

    return video_info