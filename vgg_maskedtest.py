import mtcnn
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

import vgg_scratch
from vgg_utils import *
from vgg_scratch import *

# Test the images using one masked image and another normal image

# Root image paths
root_path = "./content/faces_wild/"
img_path = "./content/faces_wild/lfw-deepfunneled-masked/"

# Read pairs and split into matched_pairs and mismatched_pairs
pairs = pd.read_csv(root_path + "pairs.csv")
pairs = pairs.rename(columns={"name": "name1", "Unnamed: 3": "name2"})
matched_pairs = pairs[pairs["name2"].isnull()].drop("name2", axis=1)
matched_pairs = matched_pairs.rename(columns={"imagenum1": "path1", "imagenum2": "path2"})
mismatched_pairs = pairs[pairs["name2"].notnull()]
mismatched_pairs = mismatched_pairs.rename(columns={"imagenum1": "path1", "imagenum2": "name2", "name2": "path2"})

# Format paths
matched_pairs["path1"] = matched_pairs["path1"].astype(int)
matched_pairs["path2"] = matched_pairs["path2"].astype(int)
mismatched_pairs["path2"] = mismatched_pairs["path2"].astype(int)

# Replace integer with image paths, path2 will be a masked image
matched_pairs["path1"] = img_path + matched_pairs["name1"] + "/" + matched_pairs["name1"] + "_" + matched_pairs[
    "path1"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"
matched_pairs["path2"] = img_path + matched_pairs["name1"] + "/" + matched_pairs["name1"] + "_" + matched_pairs[
    "path2"].apply(lambda x: '{0:0>4}'.format(x)) + "_surgical.jpg"
mismatched_pairs["path1"] = img_path + mismatched_pairs["name1"] + "/" + mismatched_pairs["name1"] + "_" + \
                            mismatched_pairs["path1"].apply(lambda x: '{0:0>4}'.format(x)) + ".jpg"
mismatched_pairs["path2"] = img_path + mismatched_pairs["name2"] + "/" + mismatched_pairs["name2"] + "_" + \
                            mismatched_pairs["path2"].apply(lambda x: '{0:0>4}'.format(x)) + "_surgical.jpg"

# Split into train and test
matched_pairs_train, matched_pairs_test = train_test_split(matched_pairs, test_size=0.2, random_state=42)
mismatched_pairs_train, mismatched_pairs_test = train_test_split(mismatched_pairs, test_size=0.2, random_state=42)

# Create train and test sets
train = pd.concat([matched_pairs_train, mismatched_pairs_train])
test = pd.concat([matched_pairs_test, mismatched_pairs_test])


# Evaluate model
def matched_pair_evaluate(matched_pairs):
    model = vgg_scratch.define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    scores = []
    same = []
    for i in range(len(matched_pairs)):
        filenames = [matched_pairs.iloc[i]["path1"], matched_pairs.iloc[i]["path2"]]
        print(filenames)
        embeddings = get_embeddings(filenames, detector, vgg_descriptor)
        if embeddings is not None:
            score = is_match(embeddings[0], embeddings[1])
            scores.append(score)
            same.append((lambda a: a <= 0.5)(score))
        else:
            print("No face detected")
            scores.append(None)
            same.append(None)
    matched_pairs["score"] = scores
    matched_pairs["same"] = same
    # Calculate accuracy
    accuracy = len(matched_pairs[matched_pairs["same"] == True]) / len(matched_pairs)
    print("Accuracy: ", accuracy)
    # Save to csv
    matched_pairs.to_csv(root_path + "matched_pairs_eval_masked.csv", index=False)

def mismatched_pair_evaluate(mismatched_pairs):
    model = vgg_scratch.define_model()
    vgg_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    detector = mtcnn.MTCNN()
    scores = []
    same = []
    for i in range(len(mismatched_pairs)):
        filenames = [mismatched_pairs.iloc[i]["path1"], mismatched_pairs.iloc[i]["path2"]]
        print(filenames)
        embeddings = get_embeddings(filenames, detector, vgg_descriptor)
        if embeddings is not None:
            score = is_match(embeddings[0], embeddings[1])
            scores.append(score)
            same.append((lambda a: a <= 0.5)(score))
        else:
            print("No face detected")
            scores.append(None)
            same.append(None)
    mismatched_pairs["score"] = scores
    mismatched_pairs["same"] = same
    # Calculate accuracy
    accuracy = len(mismatched_pairs[mismatched_pairs["same"] == False]) / len(mismatched_pairs)
    print("Accuracy: ", accuracy)
    # Save to csv
    mismatched_pairs.to_csv(root_path + "mismatched_pairs_eval_masked.csv", index=False)

# faces = extract_faces("./content/mfr2/AdrianDunbar/AdrianDunbar_0002.png", mtcnn.MTCNN())
# draw_faces("./content/mfr2/AdrianDunbar/AdrianDunbar_0002.png", faces)
matched_pair_evaluate(matched_pairs_test)
mismatched_pair_evaluate(mismatched_pairs_test)
