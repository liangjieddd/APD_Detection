import numpy as np


def _print_detection_eval_metrics(coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    _cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _classes = {
            ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
        }
    def class_name(cid):
        cls = {1:'BDD',2:'CH',3:'JP',4:'JWLD',5:'LD',6:'PL',7:'QK',8:'QP',9:'ZD',10:'ZS'}
        cls_name = cls[cid]
        # cat    = _coco.loadCats([cat_id])[0]
        return cls_name

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
        coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    # print("")
    print('MAP:{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(_classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        # cat_name  = db.class_name(cls_ind)
        # print(cat_name)
        cat_name = class_name(cls)
        # print(cat_name+":")
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind, 0, 2]
        ap = np.mean(precision[precision > -1])
        print(cat_name + ':{:.1f}'.format(100 * ap))
