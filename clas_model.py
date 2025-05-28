from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
import cmath
# OMP_NUM_THREADS=1
os.environ['OMP_NUM_THREADS'] = '1'

# Determine the optimal number of clusters
def determine_optimal_clusters(X, max_clusters):
    X = X.cpu().detach().numpy()
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    best_cluster_number = max_clusters - silhouette_scores[::-1].index(max(silhouette_scores))
    return best_cluster_number

def get_speed_and_angle(data, t):
    data_size = len(data)
    res = np.zeros((data_size, 2)).astype(float)  # Format: (speed, angle)
    for i in range(len(data)):
        d_x, d_y = data[i, t] - data[i, 0]
        res[i] = cmath.polar(complex(d_x, d_y))
    return res


def process_scene(cur_data_set, train_set, args):
    obs_len, pred_len = args.obs_length, args.pred_length
    cur_data_set = cur_data_set.transpose(1,0)
    cur_data = cur_data_set.cpu().detach().numpy()
    train_data = train_set.cpu().detach().numpy()
    res_dict = {}

    cur_raw_data = cur_data
    raw_data = train_data
    # last_coords = raw_data[:, obs_len-1]
    cur_speed_and_angle = get_speed_and_angle(cur_raw_data, obs_len-1)
    speed_and_angle = get_speed_and_angle(raw_data, obs_len-1)

    for i in range(len(cur_raw_data)):
        # last_coord = last_coords[i]
        cur_speed_max = cur_speed_and_angle[i][0] * 1.1
        cur_speed_min = cur_speed_and_angle[i][0] * 0.9
        nearby_ids = []
        for j in range(len(raw_data)):
            # if j == i:
            #     continue
            if speed_and_angle[j][0] == 0 and cur_speed_and_angle[i][0] == 0:
                nearby_ids.append(j)
            elif cur_speed_min < speed_and_angle[j][0] < cur_speed_max:
                angle_diff_tmp = np.abs(cur_speed_and_angle[i][1] - speed_and_angle[j][1])
                angle_diff = np.min([angle_diff_tmp, np.pi * 2 - angle_diff_tmp])
                if angle_diff < np.pi * 0.1:
                    nearby_ids.append(j)

        res_dict[i] = nearby_ids

    assert len(res_dict) == len(cur_data)

    return res_dict


def gen_prob(scene_dict, cluster_result, args):
    res_class, res_prob = [], []
    for i in range(len(scene_dict)):
        neighbor_list = scene_dict[i]
        neighbor_clusters = cluster_result[neighbor_list]
        tmp_dict = {cluster_result[i]: 1}
        for cluster in neighbor_clusters:
            tmp_dict[cluster] = tmp_dict.get(cluster, 0) + 1
        classes = np.zeros(len(tmp_dict)).astype(int)
        prob = np.zeros(len(tmp_dict)).astype(float)
        for idx, item in enumerate(tmp_dict.items()):
            classes[idx] = item[0]
            prob[idx] = item[1]
        prob /= (len(neighbor_list) + 1)
        res_class.append(classes)
        res_prob.append(prob)

    return res_class, res_prob


def get_scene_data(cur_data_set, data_set, cluster_result, args):
    scene_dict = process_scene(cur_data_set, data_set, args)
    gt_classes, gt_prob = gen_prob(scene_dict, cluster_result, args)


    data_size = len(gt_classes)
    res = np.zeros((data_size, args.cluster_num)).astype(float)
    for i in range(data_size):
        res[i][gt_classes[i]] = gt_prob[i]

    return res