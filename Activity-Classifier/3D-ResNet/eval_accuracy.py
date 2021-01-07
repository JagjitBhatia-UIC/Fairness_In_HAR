from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import json
import argparse
import genderDetector as gd
import cv2
from pathlib import Path
import os
import csv
import cvlib as cv

def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def load_ground_truth(ground_truth_path, subset):
    with ground_truth_path.open('r') as f:
        data = json.load(f)
    
     # download pre-trained model file (one-time download)
    dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
    model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

    model = load_model(model_path)


    class_labels_map = get_class_labels(data)

    ground_truth = []
    predictedGender = dict()

    for video_id, v in data['database'].items():
        if subset != v['subset']:
            continue
        this_label = v['annotations']['label']
        print(video_id)
        print('--------')
        print(v['annotations']['segment'])
    
        ground_truth.append((video_id, class_labels_map[this_label]))
        predictedGender[video_id] = predictGenderFromVideo(model, video_id, this_label)

    return ground_truth, class_labels_map, predictedGender

def predictGenderFromVideo(model, id, label):
    imgpath = '/home/jbhati3/datasets/HMDB/HMDB_jpgs/' + str(label) + '/' + str(id) + '/' + 'image_00001.jpg'
    image = cv2.imread(imgpath)
    if image is None:
        print("Could not read input image")
        exit()
    
    return gd.predictGender(model, image)


def load_result(result_path, top_k, class_labels_map):
    with result_path.open('r') as f:
        data = json.load(f)

   
    
    result = {}
    for video_id, v in data['results'].items():
        labels_and_scores = []
        for this_result in v:
            label = class_labels_map[this_result['label']]
            score = this_result['score']
            labels_and_scores.append((label, score))

        labels_and_scores.sort(key=lambda x: x[1], reverse=True)
        result[video_id] = list(zip(*labels_and_scores[:top_k]))[0]
    return result


def remove_nonexistent_ground_truth(ground_truth, result):
    exist_ground_truth = [line for line in ground_truth if line[0] in result]

    return exist_ground_truth

def countBuckets(preds):
    m, f, u = 0, 0, 0

    for key, value in preds.items():
        if value == 'man': m += 1
        elif value == 'woman': f += 1
        else: u += 1
    

    return (m, f, u)

def correctByBucket(result, ground_truth, predictedGender):
    m, f, u = 0, 0, 0

    for line in ground_truth:
        if line[1] in result[line[0]]:
            if predictedGender[line[0]] == 'man': m += 1
            elif predictedGender[line[0]] == 'woman': f += 1
            else: u += 1

    print('MALE CORRECT: ', m)
    print('FEMALE CORRECT: ', f)
    print('UNKNOWN CORRECT: ', u)
    return (m, f, u)


def evaluate(ground_truth_path, result_path, subset, top_k, ignore):
    print('load ground truth')
    ground_truth, class_labels_map, predictedGender = load_ground_truth(ground_truth_path,
                                                       subset)

    print('number of ground truth: {}'.format(len(ground_truth)))

    print('load result')
    result = load_result(result_path, top_k, class_labels_map)
    men, women, unknown = countBuckets(predictedGender)
    print("Number of males: ", men)
    print("Number of females: ", women)
    print("Number of unknown: ", unknown)

    print('number of result: {}'.format(len(result)))

    n_ground_truth = len(ground_truth)
    ground_truth = remove_nonexistent_ground_truth(ground_truth, result)
    if ignore:
        n_ground_truth = len(ground_truth)

    print('calculate top-{} accuracy'.format(top_k))
    correct = [1 if line[1] in result[line[0]] else 0 for line in ground_truth]
    m_cor, w_cor, u_cor = correctByBucket(result, ground_truth, predictedGender)

    accuracy = sum(correct) / n_ground_truth

    m_acc = m_cor/men
    w_acc = w_cor/women
    u_acc = u_cor/unknown

    print('top-{} accuracy: {}'.format(top_k, accuracy))

    print('ACCURACY BY BUCKET')
    print('MALE: ', m_acc)
    print('FEMALE: ', w_acc)
    print('OTHER/UNKNOWN: ', u_acc)

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth_path', type=Path)
    parser.add_argument('result_path', type=Path)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--save', action='store_true')
    parser.add_argument(
        '--ignore',
        action='store_true',
        help='ignore nonexistent videos in result')

    args = parser.parse_args()

    accuracy = evaluate(args.ground_truth_path, args.result_path, args.subset,
                        args.k, args.ignore)

    if args.save:
        with (args.result_path.parent / 'top{}.txt'.format(
                args.k)).iopen('w') as f:
            f.write(str(iaccuracy))
