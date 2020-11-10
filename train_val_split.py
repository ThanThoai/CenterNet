import json
from sklearn.model_selection import train_test_split


print("Starting")
path_train_json_input = 'train_traffic_sign_dataset_root.json'
with open(path_train_json_input, 'rb') as f:
    data = json.load(f)

val_size = 0.1
list_img = [img['id'] for img in data['images']]
annotations = data['annotations']
img_train, img_val = train_test_split(list_img, test_size = val_size, random_state = 42)
train_result = {
    'info' : data['info']
    'images' : [],
    'annotations' : [], 
    'categories' : data['categories']
}
val_result = {
    'info' : data['info']
    'images' : [],
    'annotations' : [], 
    'categories' : data['categories']
}

for i, img in list_img:
    images = data['image'][i]
    anno   = []
    for a in annotations:
        if a['id'] == img:
            anno.append(a)
    if img in img_train:
        train_result['images'].append(images)
        train_result['annotations'] += anno

    else:
        val_result['images'].append(images)
        val_result['annotations'] += anno

json.dump(train_result, open('train_traffic_sign_dataset.json', 'w+'))
json.dump(val_result, open('val_traffic_sign_dataset.json', 'w+'))
print("Finished!")






