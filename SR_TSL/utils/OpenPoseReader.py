from pathlib import Path
import json

def json_pack(snippets_dir, frame_width=1, frame_height=1,is3D=False ,label='unknown', label_index=-1,pattern='*ts.json'):
    sequence_info = []
    p = Path(snippets_dir)

    for path in p.glob(pattern):
        json_path = str(path)
        #print("read:",path)
        frame_id = int(path.stem.split('_key')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []
        for person in data['people']:
            coordinates = []
            skeleton = {}
            if is3D:

                keypoints = person['pose_keypoints_3d']
                if len(keypoints)!=100:
                    keypoints=[0 for _ in range(100)]
                lefthand=person['hand_left_keypoints_3d']
                if len(lefthand)!=84:
                    lefthand=[0 for _ in range(84)]
                righthand = person['hand_right_keypoints_3d']
                if len(righthand)!=84:
                    righthand=[0 for _ in range(84)]
                for i in range(0, 100, 4):
                    coordinates .append( [keypoints[i] , keypoints[i + 1] ,keypoints[i+2],keypoints[i + 3]])

                for i in range(0, 84, 4):
                    coordinates .append([lefthand[i] , lefthand[i + 1] ,lefthand[i+2],lefthand[i + 3]])

                for i in range(0, 84, 4):
                    coordinates .append([righthand[i] , righthand[i + 1] ,righthand[i+2],righthand[i + 3]])


            else:

                keypoints = person['pose_keypoints_2d']
                if len(keypoints)!=75:
                    keypoints=[0 for _ in range(75)]
                lefthand = person['hand_left_keypoints_2d']
                if len(lefthand)!=63:
                    lefthand=[0 for _ in range(63)]
                righthand = person['hand_right_keypoints_2d']
                if len(righthand) != 63:
                    righthand = [0 for _ in range(63)]

                for i in range(0, len(keypoints), 3):
                    coordinates .append( [keypoints[i] / frame_width, keypoints[i + 1] / frame_height,keypoints[i + 2]])

                for i in range(0, 63, 3):
                    coordinates .append([lefthand[i]/frame_width , lefthand[i + 1] /frame_height,lefthand[i + 2]])

                for i in range(0, 63, 3):
                    coordinates .append([righthand[i] /frame_width, righthand[i + 1] /frame_height,righthand[i + 2]])


            skeleton['pose'] = coordinates

            skeletons += [skeleton]
        frame_data['skeleton'] = skeletons
        frame_data['is3D']=is3D
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    print("read done")
    return video_info