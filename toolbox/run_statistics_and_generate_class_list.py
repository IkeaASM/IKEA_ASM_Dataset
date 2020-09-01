# this script runs the statistics on the labeled data (e.g. how many furniture examples? how many of each action? )
# it also creates the action class list file that only includes actions which have enough examples (above threshold)

import sqlite3
import os

output_path = './dataset_indexing_files'
include_threshold = 20  # number of examples to include the action class in the dataset
if not os.path.exists(output_path):
    os.makedirs(output_path)
action_list_filename = os.path.join(output_path, 'atomic_action_list.txt')
action_object_relation_filename = os.path.join(output_path, 'action_object_relation_list.txt')


db = sqlite3.connect('ikea_annotation_db_full')
db.row_factory = sqlite3.Row
cursor_s = db.cursor()

cursor_s.execute('''SELECT id, video_name, video_path, furniture, camera, annotated \
                    FROM videos \
                    WHERE annotated = True ''')

# count the annotated furniture types
furniture_count = {'Lack TV Bench': 0, 'Lack Coffe Table':0, 'Lack Side Table':0, 'Kallax Shelf Drawer':0}
for row in cursor_s:
    furniture_count[row['furniture']] += 1

# get list of atomic actions
cursor_s.execute('''SELECT * FROM atomic_actions''')
actions_list = []

for row in cursor_s:
    actions_list.append(row['atomic_action'])
actions_list.pop(0)

# get list of objects
cursor_s.execute('''SELECT * FROM objects''')
objects_list = []
for row in cursor_s:
    objects_list.append(row['object'])
objects_list.pop(0) # first object is blank
n_objects = len(objects_list)

# reconstruct the action list to contain all possible action - object options
compound_actions_idx = []
final_actions_list = []
action_counter = {}
action_object_relation_list = []
for i, action in enumerate(actions_list):
    if "..." in action:
        compound_actions_idx.append(i + 2)
        for j, object in enumerate(objects_list):
            compound_action = action[:-4].strip() + ' ' + object
            final_actions_list.append(compound_action)
            action_counter[compound_action] = 0
            action_object_relation_list.append([i+2, j+2])  # +1 table index starts from 1, +1 first row is empty
    else:
        final_actions_list.append(action)
        action_counter[action] = 0
        action_object_relation_list.append([i + 2, 1])

# count the different action instances
cursor_s.execute('''SELECT * FROM annotations''')
for row in cursor_s:
    action = row['action_description']
    if row['atomic_action_id'] in compound_actions_idx:
        compound_action = action[:-4].strip() + ' ' + row['object_name']
        action_counter[compound_action] += 1
        pass
    else:
        action_counter[row['action_description']] += 1

# write the dataset statistics
action_class_counter = 0
with open('dataset_stats.txt', 'w') as file:
    file.write('Assembly class statistics\n')
    for key in furniture_count:
        file.write(key + ' ' + str(furniture_count[key]) + '\n')
    file.write('\nAtomic action statistics\n')
    for key in action_counter:
        file.write(key + ' ' + str(action_counter[key]) + '\n')
        action_class_counter+=1
print('Total number of action classes : {}'.format(action_class_counter))

# write the action list and action_object relation
final_action_class_counter = 0
with open(action_list_filename, 'w') as file:
    with open(action_object_relation_filename, 'w') as relation_file:
        for i, action in enumerate(final_actions_list):
            if action_counter[action] > include_threshold:
                file.write(action + '\n')
                relation_file.write(str(action_object_relation_list[i][0]) + ' ' + str(action_object_relation_list[i][1]) + '\n')
                final_action_class_counter+=1
print('Final number of action classes : {}'.format(final_action_class_counter))

print(furniture_count)
print(action_counter)
# print(compund_actions_counter)