import numpy as np


# class Evaluator(object):
#     def __init__(self, num_class):
#         self.num_class = num_class
#         self.confusion_matrix = np.zeros((self.num_class,) * 2)
#         self.eps = 1e-8

#     def get_tp_fp_tn_fn(self):
#         tp = np.diag(self.confusion_matrix)
#         fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
#         fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
#         tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
#         return tp, fp, tn, fn

#     def Precision(self):
#         tp, fp, tn, fn = self.get_tp_fp_tn_fn()
#         precision = tp / (tp + fp)
#         return precision

#     def Recall(self):
#         tp, fp, tn, fn = self.get_tp_fp_tn_fn()
#         recall = tp / (tp + fn)
#         return recall

#     def F1(self):
#         tp, fp, tn, fn = self.get_tp_fp_tn_fn()
#         Precision = tp / (tp + fp)
#         Recall = tp / (tp + fn)
#         F1 = (2.0 * Precision * Recall) / (Precision + Recall)
#         return F1

#     def OA(self):
#         OA = np.diag(self.confusion_matrix).sum() / (
#             self.confusion_matrix.sum() + self.eps
#         )
#         return OA

#     def Intersection_over_Union(self):
#         tp, fp, tn, fn = self.get_tp_fp_tn_fn()
#         IoU = tp / (tp + fn + fp)
#         return IoU

#     def Dice(self):
#         tp, fp, tn, fn = self.get_tp_fp_tn_fn()
#         Dice = 2 * tp / ((tp + fp) + (tp + fn))
#         return Dice

#     def Pixel_Accuracy_Class(self):
#         #         TP                                  TP+FP
#         Acc = np.diag(self.confusion_matrix) / (
#             self.confusion_matrix.sum(axis=0) + self.eps
#         )
#         return Acc

#     def Frequency_Weighted_Intersection_over_Union(self):
#         freq = np.sum(self.confusion_matrix, axis=1) / (
#             np.sum(self.confusion_matrix) + self.eps
#         )
#         iou = self.Intersection_over_Union()
#         FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
#         return FWIoU

#     def _generate_matrix(self, gt_image, pre_image):
#         mask = (gt_image >= 0) & (gt_image < self.num_class)
#         label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
#         count = np.bincount(label, minlength=self.num_class**2)
#         confusion_matrix = count.reshape(self.num_class, self.num_class)
#         return confusion_matrix

#     def add_batch(self, gt_image, pre_image):
#         assert (
#             gt_image.shape == pre_image.shape
#         ), "pre_image shape {}, gt_image shape {}".format(
#             pre_image.shape, gt_image.shape
#         )
#         self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

#     def reset(self):
#         self.confusion_matrix = np.zeros((self.num_class,) * 2)


class Evaluator(object):
    def __init__(self, num_class, eps=1e-8, ignore_index=None):
        self.num_class = num_class
        self.eps = eps
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros(
            (self.num_class, self.num_class), dtype=np.int64
        )

    def get_tp_fp_tn_fn(self):
        cm = self.confusion_matrix
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        total = cm.sum().astype(np.float64)
        tn = total - tp - fp - fn
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, _, _ = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + self.eps)

    def Recall(self):
        tp, _, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fn + self.eps)

    def F1(self):
        p = self.Precision()
        r = self.Recall()
        return 2.0 * p * r / (p + r + self.eps)

    def OA(self):
        return np.diag(self.confusion_matrix).sum() / (
            self.confusion_matrix.sum() + self.eps
        )

    def Intersection_over_Union(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return tp / (tp + fp + fn + self.eps)

    def Dice(self):
        tp, fp, _, fn = self.get_tp_fp_tn_fn()
        return 2 * tp / (2 * tp + fp + fn + self.eps)

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=0) + self.eps
        )
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (
            np.sum(self.confusion_matrix) + self.eps
        )
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def mIoU(self):
        iou = self.Intersection_over_Union()
        freq = self.confusion_matrix.sum(axis=1)
        valid = freq > 0
        return float(np.mean(iou[valid])) if np.any(valid) else 0.0

    def mF1(self):
        f1 = self.F1()
        freq = self.confusion_matrix.sum(axis=1)
        valid = freq > 0
        return float(np.mean(f1[valid])) if np.any(valid) else 0.0

    def _generate_matrix(self, gt_image, pre_image):
        gt = gt_image.astype(np.int64)
        pr = pre_image.astype(np.int64)

        if self.ignore_index is not None:
            mask = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_class)
        else:
            mask = (gt >= 0) & (gt < self.num_class)

        gt = gt[mask]
        pr = pr[mask]

        label = self.num_class * gt + pr
        count = np.bincount(label, minlength=self.num_class**2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert (
            gt_image.shape == pre_image.shape
        ), f"pre_image shape {pre_image.shape}, gt_image shape {gt_image.shape}"
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix[:] = 0


if __name__ == "__main__":

    gt = np.array([[0, 2, 1], [1, 2, 1], [1, 0, 1]])

    pre = np.array([[0, 1, 1], [2, 0, 1], [1, 1, 1]])

    eval = Evaluator(num_class=3)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
