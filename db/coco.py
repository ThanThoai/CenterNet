import sys
sys.path.insert(0, "data/coco/PythonAPI/")

import os
import json
import numpy as np
import pickle

from tqdm import tqdm
from db.detection import DETECTION
from config import system_configs
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class MSCOCO(DETECTION):
    def __init__(self, db_config, split):
        super(MSCOCO, self).__init__(db_config)
        data_dir   = system_configs.data_dir
        result_dir = system_configs.result_dir
        cache_dir  = system_configs.cache_dir

        self._split = split
        self._dataset = {
            "trainval": "train",
            "val"     : "val"
        }[self._split]
        
        self._zalo_dir = os.path.join(data_dir, "za_traffic_2020/traffic_train")

        # self._label_dir  = os.path.join(self._zalo_dir, "annotations")
        self._label_file = os.path.join(self._zalo_dir, f"{self._dataset}_traffic_sign.json")
        # self._label_file = self._label_file.format(self._dataset)

        self._image_dir  = os.path.join(self._zalo_dir, "images")
        self._image_file = os.path.join(self._image_dir, "{}")

        self._data = "Zalo"
        self._mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std  = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._cat_ids = [
            1, 2, 3, 4, 5, 6
        ]
        self._classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(self._cat_ids)
        }
        self._zalo_to_class_map = {
            value: key for key, value in self._classes.items()
        }

        self._cache_file = os.path.join(cache_dir, "zalo_traffic_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_zalo_data() 

    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)

    def _load_zalo_data(self):
        self._zalo = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        zalo_ids = self._zalo.getImgIds()
        eval_ids = {
            self._zalo.loadImgs(zalo_id)[0]["file_name"]: zalo_id
            for zalo_id in zalo_ids
        }

        self._zalo_categories = data["categories"]
        self._zalo_eval_ids   = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat    = self._zalo.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._zalo    = COCO(self._label_file)
        self._cat_ids = self._zalo.getCatIds()

        zalo_image_ids = self._zalo.getImgIds()

        self._image_ids = [
            self._zalo.loadImgs(img_id)[0]["file_name"] 
            for img_id in zalo_image_ids
        ]
        self._detections = {}
        for ind, (zalo_image_id, image_id) in enumerate(tqdm(zip(zalo_image_ids, self._image_ids))):
            image      = self._zalo.loadImgs(zalo_image_id)[0]
            bboxes     = []
            categories = []

            for cat_id in self._cat_ids:
                annotation_ids = self._zalo.getAnnIds(imgIds=image["id"], catIds=cat_id)
                annotations    = self._zalo.loadAnns(annotation_ids)
                category       = self._zalo_to_class_map[cat_id]
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bbox[[2, 3]] += bbox[[0, 1]]
                    bboxes.append(bbox)

                    categories.append(category)

            bboxes     = np.array(bboxes, dtype=float)
            categories = np.array(categories, dtype=float)
            if bboxes.size == 0 or categories.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                self._detections[image_id] = np.hstack((bboxes, categories[:, None]))

    def detections(self, ind):
        image_id = self._image_ids[ind]
        detections = self._detections[image_id]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_to_zalo(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            zalo_id = self._zalo_eval_ids[image_id]
            for cls_ind in all_bboxes[image_id]:
                category_id = self._classes[cls_ind]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]

                    score = bbox[4]
                    bbox  = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": zalo_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": float("{:.2f}".format(score))
                    }

                    detections.append(detection)
        return detections

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        zalo = self._zalo if gt_json is None else COCO(gt_json)

        eval_ids = [self._zalo_eval_ids[image_id] for image_id in image_ids]
        cat_ids  = [self._classes[cls_id] for cls_id in cls_ids]

        zalo_dets = zalo.loadRes(result_json)
        zalo_eval = COCOeval(zalo, zalo_dets, "bbox")
        zalo_eval.params.imgIds = eval_ids
        zalo_eval.params.catIds = cat_ids
        zalo_eval.evaluate()
        zalo_eval.accumulate()
        zalo_eval.summarize()
        zalo_eval.evaluate_fd()
        zalo_eval.accumulate_fd()
        zalo_eval.summarize_fd()
        return zalo_eval.stats[0], zalo_eval.stats[12:]
