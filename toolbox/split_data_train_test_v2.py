# perform an environment based data split
import sqlite3
import os

def parse_video_name(video_name):
    data = video_name.split('_')

    person_id = int(data[0])
    color = data[1]
    asm_conf = data[2]
    calibration = int(data[3])
    room = int(data[4])
    date_time = data[5:]
    return person_id, color, asm_conf, calibration, room, date_time


def handle_devices(train_file_list, test_file_list):
    # remove the device suffix and keep just the directory path for the scans
    new_train_file_list = []
    new_test_file_list = []
    for dir_name in train_file_list:
        new_train_file_list.append(dir_name[:-12]) # remove 'dev3/images'
    for dir_name in test_file_list:
        new_test_file_list.append(dir_name[:-12])  # remove 'dev3/images'
    return new_train_file_list, new_test_file_list


def write_set_to_file(set_filename, set_list):
    n_items = len(set_list)
    print('Writing {} scans to {}'.format(n_items, set_filename))
    with open(set_filename, 'w') as f:
        for item in set_list:
            f.write("%s\n" % item)

output_path = './dataset_indexing_files'
if not os.path.exists(output_path):
    os.makedirs(output_path)
trainset_filename = os.path.join(output_path, 'train_cross_env.txt')
testset_filename = os.path.join(output_path, 'test_cross_env.txt')
annotatedset_filename = os.path.join(output_path, 'ikea_annotated.txt')
allset_filename = os.path.join(output_path, 'ikea_all.txt')

env_dict = {"env1": [1, 2], "env2": [4, 5], "env3": [6, 7], "env4": [8], "env5": [9, 10, 11]}
calibration_room_equivalence = [[1, 2], [4, 5], [6, 7], [9, 10, 11]]
test_calibrations = env_dict["env2"] + env_dict["env4"]  #[4, 5, 8] # spcify the test environments

db = sqlite3.connect('ikea_annotation_db_full')
db.row_factory = sqlite3.Row

cursor_s = db.cursor()
cursor_s.execute('''SELECT id, video_name, video_path, furniture, camera, annotated \
                    FROM videos''')
# fetch all scans
total_scene_counter = [0] * 12
allscans_file_list = []
for row in cursor_s:
    video_path = row['video_path']
    if 'special' not in row['video_name']: #and 'Allstar' not in row['video_name']:  # ignore the special test folder
        person_id, color, asm_conf, calibration, room, date_time = parse_video_name(row['video_name'])
        total_scene_counter[calibration] += 1
        allscans_file_list.append(video_path)

cursor_s.execute('''SELECT id, video_name, video_path, furniture, camera, annotated \
                    FROM videos \
                    WHERE annotated = True AND video_path LIKE '%dev3%' ''')


scene_counter = [0] * 12
cursor_s = list(cursor_s)

# fetch all possible annotated  scans
annotated_file_list = []
for row in cursor_s:
    video_path = row['video_path']
    if 'special' not in row['video_name']: #and 'Allstar' not in row['video_name']:  # ignore the special test folder
        person_id, color, asm_conf, calibration, room, date_time = parse_video_name(row['video_name'])
        scene_counter[calibration] += 1
        annotated_file_list.append(video_path)

# fetch all possible test scans
test_file_list = []
test_persons = []
for row in cursor_s:
    video_path = row['video_path']
    furniture = row['furniture']
    if 'special' not in row['video_name']: #and 'Allstar' not in row['video_name']: # ignore the special test folder
        person_id, color, asm_conf, calibration, room, date_time = parse_video_name(row['video_name'])
        if calibration in test_calibrations:
            test_file_list.append(video_path)
            if not person_id in test_persons:
                test_persons.append(person_id)

# fetch all possible trainset scans
train_file_list = []
train_persons = []
for row in cursor_s:
    video_path = row['video_path']
    furniture = row['furniture']
    if 'special' not in row['video_name']: #and 'Allstar' not in row['video_name']: # ignore the special test folder
        person_id, color, asm_conf, calibration, room, date_time = parse_video_name(row['video_name'])

        if calibration not in test_calibrations:
            train_file_list.append(video_path)
            if not person_id in train_persons:
                train_persons.append(person_id)


train_file_list, test_file_list = handle_devices(train_file_list, test_file_list)

print("Note scene equivalence for :{}".format(calibration_room_equivalence))
print("number of total scans in each scene: {}".format(total_scene_counter))
print("number of annotated scans in each scene: {}".format(scene_counter))

train_ratio = len(train_file_list) / (len(train_file_list)+ len(test_file_list))
test_ratio = 1 - train_ratio

print("Split info: train {}, test {}, ratio {}/{}".format(len(train_file_list), len(test_file_list), train_ratio, test_ratio))
print("Train persons: {}".format(sorted(train_persons)))
print("Test persons: {}".format(sorted(test_persons)))
unique_test_persons = (set(test_persons)) - (set(test_persons) & set(train_persons))
print("unique test persons: {}".format(unique_test_persons))

write_set_to_file(allset_filename, allscans_file_list)
write_set_to_file(annotatedset_filename, annotated_file_list)
write_set_to_file(trainset_filename, train_file_list)
write_set_to_file(testset_filename, test_file_list)
