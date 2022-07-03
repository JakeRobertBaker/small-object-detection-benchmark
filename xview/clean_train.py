import json
import copy

with open('data/xview/coco/train.json') as train_file:
    data = json.load(train_file)
    
clean_annotations = []

for ann in data['annotations']:
    x,y,w,h = ann['bbox']
    if (x>=0 and y>=0 and w>=0 and h>=0):
        clean_annotations.append(ann)
        
clean_data = copy.deepcopy(data)

clean_data['annotations'] = clean_annotations

with open('data/xview/coco/train_clean.json', 'w') as outfile:
    json.dump(clean_data, outfile)