# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import numpy as np
from PIL import Image
import imgviz

from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules


def eval_iou(pred_label, raw_label, num_classes, file_name):

    area_intersect = np.zeros((num_classes,), dtype='float')
    area_pred_label = np.zeros((num_classes,), dtype='float')
    area_label = np.zeros((num_classes,), dtype='float')

    for i in range(1, num_classes+1):
        if (num_classes == 2 and i == 1) or (num_classes == 7 and i == 2):
            pred = np.zeros(pred_label[i].shape, dtype='int')
            pred[pred_label[i - 1] > 0.5] = 1
            pred[pred_label[i] > 0.5] = 1
            label = np.zeros(raw_label.shape, dtype='int')
            label[raw_label==i] = 1
            label[raw_label==i+1] = 1
        else:
            pred = pred_label[i - 1] > 0.5
            label = raw_label == i
        area_intersect[i-1] = np.sum(label & pred)
        area_pred_label[i-1] = np.sum(pred)
        area_label[i-1] = np.sum(label)
    
    area_union = area_pred_label + area_label - area_intersect
    # dice = 2 * area_intersect / (area_pred_label + area_label)
    # print('{:<20} {:.2f}'.format(file_name,  np.round(dice[5] * 100, 2)))
    return area_intersect, area_union, area_pred_label, area_label


def total_iou(results, gt_seg_maps, num_classes, img_list):
    total_pred = np.zeros((num_classes,), dtype='float')
    total_tp = np.zeros((num_classes,), dtype='float')
    total_label = np.zeros((num_classes,), dtype='float')
    total_union = np.zeros((num_classes,), dtype='float')

    for result, gt_seg_map, file_name in zip(results, gt_seg_maps, img_list):
        tp, union, pred, label = eval_iou(result, gt_seg_map, num_classes, file_name)
        total_tp += tp
        total_union += union
        total_pred += pred
        total_label += label

    iou = total_tp / total_union
    dice = 2 * total_tp / (total_pred + total_label)
    precision = total_tp / total_pred
    recall = total_tp / total_label
    return iou, dice, precision, recall

# ../../data/FOVCrop-padding/DDR-FOVCrop-padding/test/images
# ../../data/FOVCrop-padding/DDR-FOVCrop-padding/test/ann

output_dir = 'refuge'
os.makedirs(output_dir, exist_ok=True)

def main():
    parser = ArgumentParser()
    parser.add_argument('img_path', help='Image path')
    parser.add_argument('ann_path', help='Ann path')
    parser.add_argument('num_classes', help='num_classes')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    results = []
    anns = []
    img_list = os.listdir(args.img_path)
    img_list.sort()
    for file_name in img_list:
        # file_name = 'IDRiD_56.jpg'
        img = cv2.imread(os.path.join(args.img_path, file_name))
        result = inference_model(model, img).seg_logits.cpu().data
        # for i in range(result.shape[0]):
        #     ii = (result[i]>0.2).int().numpy().astype(np.uint8)
        #     ii[ii==1] = 255
        #     cv2.imwrite(f'{i}.png', ii)
        # output = result.argmax(0) + 1
        # output[(result<0.2).all(dim=0)] = 0
        # ii = (result[0]>0.5).int().numpy().astype(np.uint8)
        # ii[ii==1] = 255
        # cv2.imwrite(os.path.join(output_dir, file_name.split('.')[0] + '.png'), ii)
        output = result.numpy()
        # OD
        # label_OD = np.zeros(output.shape[-2:], dtype=np.uint8)
        # label_OD[output[1]>0.5] = 255
        # label_OD[output[2]>0.5] =255
        # cv2.imwrite(os.path.join(output_dir, file_name.split('.')[0] + '.png'), label_OD)
        # lesion
        # label_lesion = np.argmax(output[3:7], axis=0) + 1
        # label_lesion[(output[3:7] < 0.5).all(axis=0)] = 0
        # label_pil = Image.fromarray(label_lesion.astype(np.uint8), mode="P")
        # colormap = imgviz.label_colormap()
        # label_pil.putpalette(colormap)
        # label_pil.save(os.path.join(output_dir, file_name.split('.')[0] + '.png'))
        results.append(output)
        ann = np.array(Image.open(os.path.join(args.ann_path, file_name.split('.')[0] + '.png')))
        # ann[ann==1] = 2
        anns.append(ann)
        # print(file_name)
        # iou, dice, precision, recall = total_iou([output], [ann], int(args.num_classes), [file_name])
        # print(file_name, dice)
        # break

    iou, dice, precision, recall = total_iou(results, anns, int(args.num_classes), img_list)
    iou = np.round(iou*100, 2)
    dice = np.round(dice*100, 2)
    precision = np.round(precision*100, 2)
    recall = np.round(recall*100, 2)
    print(iou, dice, precision, recall)


if __name__ == '__main__':
    main()
