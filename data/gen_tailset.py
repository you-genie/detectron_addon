import json
from tqdm import tqdm

with open("train.json", "r") as json_file:
    dataset = json.load(json_file)
    tail_set = {
            'info': {}, 'licenses': {}, 'images': [], 'categories': dataset['categories'], 'annotations': []
    }
    images = dataset['images']
    annos = dataset['annotations']

    image_id = set()
    for anno in tqdm(annos):
        cat = anno['category_id']
        if cat in [6, 3, 11, 13]:
            tail_set['annotations'].append(anno)
            image_id.add(anno['image_id'])

    for img in tqdm(images):
        if img['id'] in image_id:
            tail_set['images'].append(img)

    print(f"filtered         images: {len(tail_set['images'])}, annos: {len(tail_set['annotations'])}")
    with open('train_tail.json', 'w') as tail_file:
        json.dump(tail_set, tail_file)

