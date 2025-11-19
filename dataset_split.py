import os
import json
import scipy.io as sio
from utils import load_config

cfg = load_config()

#Get dataset path and class names
def dataset_preprocess(dataset_name):

    main_dir  = cfg['dataset']['root']

    if dataset_name == "cub":
        dataset_dir = os.path.join(main_dir, "CUB_200_2011")

    elif dataset_name == "flower":
        dataset_dir = os.path.join(main_dir, "flowers-102")

    elif dataset_name == "food":
        dataset_dir = os.path.join(main_dir, "food-101")

    elif dataset_name == "pet":
        dataset_dir = os.path.join(main_dir, "pet_37")

    elif dataset_name == "car":
        dataset_dir = os.path.join(main_dir, "car_196")

    elif dataset_name == "dog":
        dataset_dir = os.path.join(main_dir, "dogs_120")

    elif dataset_name == "aircraft":
        dataset_dir = os.path.join(main_dir, "fgvc-aircraft-2013b")

    elif dataset_name == "sun":
        dataset_dir = os.path.join(main_dir, "SUN397")

    else:
        print("Unknown dataset")
        return

    with open(os.path.join(dataset_dir, f"{dataset_name}_class_names.txt")) as f:
        class_names = [line.strip() for line in f.readlines()]

    return dataset_dir, class_names

#Split the test set and save as json file
def sample_test(dataset_name, save_path="./test_sample"):

    dataset_dir, class_names = dataset_preprocess(dataset_name)
    samples = []

    if dataset_name == "flower":
        labels = sio.loadmat(os.path.join(dataset_dir, 'imagelabels.mat'))['labels'][0]
        test_ids = sio.loadmat(os.path.join(dataset_dir, 'setid.mat'))['tstid'][0]

        for idx in test_ids:
            image_path = os.path.join(dataset_dir, 'jpg', f'image_{idx:05d}.jpg')
            label_index = labels[idx - 1] - 1
            label_name = class_names[label_index]
            samples.append({
                "image_path": image_path,
                "label_index": int(label_index),
                "label_name": label_name
            })

    elif dataset_name == "pet":
        with open(os.path.join(dataset_dir, "annotations", "test.txt")) as f:
            lines = f.readlines()
            for line in lines:
                tokens = line.strip().split()
                img_name = tokens[0] + ".jpg"
                label = int(tokens[1]) - 1
                samples.append({
                    "image_path":os.path.join(dataset_dir,"images",img_name),
                    "label_index": label,
                    "label_name": class_names[label]
                })

    elif dataset_name == "cub":

        image_paths = []
        with open(os.path.join(dataset_dir,"images.txt")) as f:
            for line in f:
                image_path = line.strip().split()[1]
                image_paths.append(os.path.join(dataset_dir,"images",image_path))

        labels = []
        with open(os.path.join(dataset_dir,"image_class_labels.txt")) as f:
            for line in f:
                label = line.strip().split()[1]
                labels.append(int(label)-1) #0-based

        test_split = []
        with open(os.path.join(dataset_dir,"train_test_split.txt")) as f:
            for line in f:
                is_test = line.strip().split()[1]
                test_split.append(int(is_test))

        for image_path, label, is_test in zip(image_paths, labels, test_split):
            if is_test==0:
                samples.append({
                    "image_path": image_path,
                    "label_index": label,
                    "label_name": class_names[label]
                })

    elif dataset_name == "food":
        images_dir = os.path.join(dataset_dir, "images")
        class_to_index = {}
        i = 0
        with open(os.path.join(dataset_dir, "meta", "classes.txt")) as f:
            for line in f:
                class_to_index[line.strip()] = i
                i = i + 1


        with open(os.path.join(dataset_dir, "meta", "test.txt")) as f:
            for line in f:
                class_name = line.strip().split("/", 1)[0]
                label_index = class_to_index[class_name]
                samples.append({
                    "image_path": os.path.join(images_dir, line.strip() + ".jpg"),
                    "label_index": label_index,
                    "label_name": class_names[label_index],
                })

    elif dataset_name == "dog":
        test_list = sio.loadmat(os.path.join(dataset_dir,"test_list.mat"))
        test_image_paths = test_list['file_list']
        test_labels = test_list['labels']
        for label, image_path in zip(test_labels, test_image_paths):
            label_index = label[0]-1
            label_name = class_names[label_index]
            image_path = os.path.join(dataset_dir,"Images",image_path[0][0])
            samples.append({
                "image_path":image_path,
                "label_index": int(label_index),
                "label_name":label_name
            })

    elif dataset_name == "car":
        annotations = sio.loadmat(os.path.join(dataset_dir,"devkit/cars_test_annos_withlabels.mat"))['annotations'][0]

        for anno in annotations:
            label_index = int(anno[4][0][0])-1
            image_name = anno[5][0]
            image_path = os.path.join(dataset_dir,"cars_test",image_name)
            label_name = class_names[label_index]

            samples.append({
                "image_path":image_path,
                "label_index":label_index,
                "label_name":label_name
            })

    elif dataset_name == "aircraft":
        class_to_index = {}
        i = 0
        for class_name in class_names:
            class_to_index[class_name] = i
            i = i + 1

        with open(os.path.join(dataset_dir, "data/images_variant_test.txt")) as f:
            for line in f.readlines():
                image_name = line.split(' ',1)[0]
                label_name = line.split(' ',1)[1].strip()
                image_path = os.path.join(dataset_dir,"data/images",image_name+".jpg")
                label_index = class_to_index[label_name]

                samples.append({
                    "image_path":image_path,
                    "label_index":label_index,
                    "label_name":label_name,
                })

    elif dataset_name == "sun":

        with open(os.path.join(dataset_dir,"Partitions/ClassName.txt")) as f:
            class_list = [line.strip() for line in f]

        class_to_index = {}
        for idx, name in enumerate(class_list):
            class_to_index[name] = idx

        #Loading only the first test split, 397 scene categories, 50 images per category.
        with open(os.path.join(dataset_dir,"Partitions/Testing_01.txt")) as f:
        #/a/abbey/sun_ajkqrqitspwywirx.jpg
            for line in f.readlines():
                image_path = line.strip()
                label = image_path.rsplit("/",1)[0]
                #print(label)
                index = class_to_index[label]
                new_label = class_names[index]
                image_path = dataset_dir+image_path
                samples.append({
                    "image_path": image_path,
                    "label_index": index,
                    "label_name": new_label,
                })
    else:
        print("Unknown dataset")
        return

    print("---------------------------------------")
    print(f"Dataset: {dataset_name}, Number of Categories: {len(class_names)}")
    print(f"Dataset Path: {dataset_dir}")
    print(f"Size of Test Samples: {len(samples)}")

    save_path = os.path.join(save_path,f"{dataset_name}_test.json")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open (save_path, "w") as f:
        json.dump(samples, f, indent=2)

if __name__ == '__main__':
    dataset_list = ["flower","cub","pet","food","aircraft","car","dog","sun"]
    for dataset_name in dataset_list:
        dataset_dir, class_names = dataset_preprocess(dataset_name)
        save_dir = cfg['dataset']['test_path']
        sample_test(dataset_name, save_dir)