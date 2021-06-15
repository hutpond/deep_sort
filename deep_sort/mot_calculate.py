import numpy as np
import math

class MotAssessment:
    """ 评估MOT结果精度, 包括MOTA, MOTP
    """
    def __init__(self, name, dis_thres):
        self.dis_thres = dis_thres
        self.gt_infos = {}
        self.boxes_info = {}

        with open(name, "r") as f:
            lines = f.readlines()
        lines = np.array([line.strip().split(",") for line in lines], dtype=float)
        image_ids = np.unique(lines[:, 0])
        for image_id in image_ids:
            image_id = int(image_id)
            image_detection = lines[lines[:, 0] == image_id]
            image_detection = image_detection[:, [1, 2, 3, 4, 5]]
            image_detection[:, 3] += (image_detection[:, 1] - 1)
            image_detection[:, 4] += (image_detection[:, 2] - 1)
            image_detection = image_detection.tolist()
            self.gt_infos[image_id] = image_detection

    def add_detections(self, img_idx, boxes):
        self.boxes_info[int(img_idx)] = boxes

    def distance(self, box, box2):
        aleft, atop, aright, abottom = box[:4]
        bleft, btop, bright, bbottom = box2[:4]
        x1 = (aleft + aright) * 0.5
        y1 = (atop + abottom) * 0.5
        x2 = (bleft + bright) * 0.5
        y2 = (btop + bbottom) * 0.5
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


    def calculate(self):
        values = []

        # MOTA
        mota = self.calculate_mota()
        values.append(mota)
        # MOTP
        motp = self.calculate_motp()
        values.append(motp)

        return values

    
    def calc_coin_values(self, first_infos, second_infos, dis_thres):
        dis_values = []
        for first_info in first_infos:
            values = []
            for second_info in second_infos:
                value = self.distance(first_info[1:], second_info[1:])
                values.append(value)
            dis_values.append(values)

        def min_fun(values):
            min_val = 1.0e10
            Rows = values.__len__()
            Cols = 0 if values.__len__() == 0 else values[0].__len__()
            position = [0, 0]
            for i in range(Rows):
                for j in range(Cols):
                    if values[i][j] < min_val:
                        position[0] = i
                        position[1] = j
                        min_val = values[i][j]

            return position

        coin_values = []
        Rows = first_infos.__len__()
        Cols = second_infos.__len__()

        while Rows > 0 and Cols > 0:
            pos = min_fun(dis_values)
            if dis_values[pos[0]][pos[1]] >= dis_thres:
                break
            for i in range(Rows):
                dis_values[i][pos[1]] = 1.0e10
            for j in range(Cols):
                dis_values[pos[0]][j] = 1.0e10
            coin_values.append(pos)

        return coin_values
        

    # Multiple Object Tracking Accuracy
    def calculate_mota(self):
        gt_sum = 0
        fn_sum = 0
        fp_sum = 0
        idsw_sum = 0
        coin_values_pre = []
        coin_values = []

        for img_index in self.boxes_info:
            boxes_info = self.boxes_info[img_index]
            gt_infos = self.gt_infos[img_index]

            # 重合框位置
            coin_values_pre.clear()
            coin_values_pre = coin_values
            coin_values = self.calc_coin_values(boxes_info, gt_infos, self.dis_thres)
            for coin_value in coin_values:
                coin_value[0] = boxes_info[coin_value[0]][0]
                coin_value[1] = gt_infos[coin_value[1]][0]

            # GT
            gt_sum += gt_infos.__len__()
            # FN 漏检个数
            fn_sum += gt_infos.__len__() - coin_values.__len__()
            # FP 误检个数
            fp_sum += boxes_info.__len__() - coin_values.__len__()
            # IDSW
            if img_index > 1:
                for coin_value in coin_values:
                    for coin_value_pre in coin_values_pre:
                        if (coin_value[0] == coin_value_pre[0] and 
                            coin_value[1] == coin_value_pre[1]):
                            continue
                        if (coin_value[0] != coin_value_pre[0] and 
                            coin_value[1] != coin_value_pre[1]):
                            continue
                        idsw_sum += 1

        # MOTA
        print(f'subitem: {fn_sum} {fp_sum} {idsw_sum}')
        return 1.0 - (fn_sum + fp_sum + idsw_sum) / gt_sum
    

    # MOTP Multiple Object Tracking Precision
    def calculate_motp(self):
        c_sum = 0
        d_sum = 0.0
        for img_index in self.boxes_info:
            boxes_info = self.boxes_info[img_index]
            gt_infos = self.gt_infos[img_index]

            # 成功与GT匹配的检测框数目
            coin_values = self.calc_coin_values(boxes_info, gt_infos, self.dis_thres)
            for coin_value in coin_values:
                d_sum += self.distance(
                    boxes_info[coin_value[0]][1:],
                    gt_infos[coin_value[1]][1:])
            c_sum += coin_values.__len__()

        return 0 if c_sum == 0 else d_sum / c_sum
