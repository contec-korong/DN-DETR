# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from libs.label_name_dict.label_dict import NAME_LABEL_MAP
# from libs.configs import cfgs
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


def voc_ap(rec, prec):
    """
    Compute VOC AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    ind = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[ind]) * mpre[ind + 1])
    return ap


def voc_eval(model, postprocessors, data_loader, device, n_pattern=0, img_size=768, iou_thr=0.3):
    """
    :param model:
    :param postprocessors:
    :param data_loader:
    :param device:
    :param n_pattern:
    :param iou_thr:
    :return:
    """
    tp = []
    fp = []
    num_target = 0
    for samples, _targets in data_loader:
        samples = samples.to(device)

        targets = []
        for _target in _targets:
            _target['boxes'][:, :4] = _target['boxes'][:, :4] * img_size
            _target['boxes'][:, 4] = _target['boxes'][:, 4] * torch.pi
            for box, label in zip(_target['boxes'], _target['labels']):
                targets.append({'boxes': box.to(device), 'labels': label.to(device)})
        num_target += len(targets)

        # with torch.cuda.amp.autocast(enabled=args.amp):
        model.eval()
        with torch.no_grad():
            outputs = postprocessors(model(samples, n_pattern)[0],
                                     target_size=768)   # [{'scores': s, 'labels': l, 'boxes': b}]

        targets_map = [0] * len(targets)
        if len(outputs):
            nd = len(outputs)  # num of detections.
            # sort by confidence
            outputs = sorted(outputs, key=lambda x: x['scores'], reverse=True)
            for i, output in enumerate(outputs):
                _tp = [0] * nd
                _fp = [0] * nd
                max_iou = -1
                max_ind = -1

                box1 = output['boxes'].detach().cpu().numpy()
                r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])  # sjhong

                for j, target in enumerate(targets):
                    if output['labels'] == target['labels']:
                        box2 = target['boxes'].detach().cpu().numpy()
                        r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])  # sjhong

                        int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                        if int_pts is not None:
                            order_pts = cv2.convexHull(int_pts, returnPoints=True)
                            int_area = cv2.contourArea(order_pts)
                            area1 = box1[2] * box1[3]
                            area2 = box2[2] * box2[3]
                            inter = int_area * 1.0 / (area1 + area2 - int_area)
                            if max_iou < inter:
                                max_iou = inter
                                max_ind = j
                # To draw recall, precision graph, get tp, fp in stack shape
                if max_iou > iou_thr:
                    if not targets_map[max_ind]:
                        _tp[i] = 1
                        targets_map[max_ind] = 1
                    else:
                        _fp[i] = 1
                else:
                    _fp[i] = 1
                tp.extend(_tp)
                fp.extend(_fp)

    # Get recall, precison and AP
    tp = np.array(tp, dtype=float)
    fp = np.array(fp, dtype=float)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_target)
    prec = tp / (tp + fp)
    ap = voc_ap(rec, prec)

    return {'recall': rec[-1] * 100, 'precesion': prec[-1] * 100, 'AP': ap * 100}

"""
def do_python_eval(test_imgid_list, test_annotation_path):
    # import matplotlib.colors as colors
    # import matplotlib.pyplot as plt

    AP_list = []
    for cls, index in NAME_LABEL_MAP.items():
        if cls == 'back_ground':
            continue
        recall, precision, AP = voc_eval(detpath=cfgs.EVALUATE_R_DIR,
                                         test_imgid_list=test_imgid_list,
                                         cls_name=cls,
                                         annopath=test_annotation_path,
                                         ovthresh=cfgs.EVAL_THRESHOLD)
        AP_list += [AP]
        print("cls : {}|| Recall: {} || Precison: {}|| AP: {}".format(cls, recall[-1], precision[-1], AP))
        # print("{}_ap: {}".format(cls, AP))
        # print("{}_recall: {}".format(cls, recall[-1]))
        # print("{}_precision: {}".format(cls, precision[-1]))
        r = np.array(recall)
        p = np.array(precision)
        F1 = 2 * r * p / (r + p + 1e-5)
        max_ind = np.argmax(F1)
        print('F1:{} P:{} R:{}'.format(F1[max_ind], p[max_ind], r[max_ind]))

        # c = colors.cnames.keys()
        # c_dark = list(filter(lambda x: x.startswith('dark'), c))
        # c = ['red', 'orange']
        # plt.axis([0, 1.2, 0, 1])
        # plt.plot(recall, precision, color=c_dark[index], label=cls)

    # plt.legend(loc='upper right')
    # plt.xlabel('R')
    # plt.ylabel('P')
    # plt.savefig('./PR_R.png')

    print("mAP is : {}".format(np.mean(AP_list)))
"""