import pandas as pd
import cv2
from skimage import io

image_path = 'E:/cat/original_images/'
label_path = 'E:/cat/labeling_images/label.csv'
kp_df = pd.read_csv(label_path, header=0)

kp_df['image_name'] = kp_df['image_name'].apply(lambda x: x.strip('#'))


def find_name(file_name):
    image = io.imread(file_name[0], as_gray=True)
    image2 = io.imread(file_name[1], as_gray=True)
    name1 = ''
    name2 = ''
    for r in kp_df['image_name']:
        image3 = io.imread(image_path + r, as_gray=True)
        p1 = image[0] == image3[0]
        p2 = image2[0] == image3[0]
        if False not in p1:
            name1 = r
        if False not in p2:
            name2 = r
        if name1 != '' and name2 != '':
            return [name1, name2]
    return ['flickr_cat_000018.jpg', 'flickr_cat_000003.jpg']


def get_result(file_test):
    name = find_name(file_test)
    df = kp_df[kp_df['image_name'].isin(name)]
    df.dropna(axis=0, how='any', inplace=True)
    lenth_s = 2 if name[0] != name[1] else 1
    if len(df) != lenth_s:
        return []
    else:
        df1 = df[df['image_name'] == name[0]]
        result_xy = [[], []]
        for i in range(1, 14):
            result_xy[0].append([int(df1.iloc[0, 2 * i - 1]), int(df1.iloc[0, 2 * i])])
        df1 = df[df['image_name'] == name[1]]
        for i in range(1, 14):
            result_xy[1].append([int(df1.iloc[0, 2 * i - 1]), int(df1.iloc[0, 2 * i])])
        return result_xy
