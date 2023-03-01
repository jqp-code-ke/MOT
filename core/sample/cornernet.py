import cv2
import math
import numpy as np
import torch

from .utils import random_crop, draw_gaussian, gaussian_radius, normalize_, color_jittering_, lighting_

def _resize_image(image, detections, size):
    detections    = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections

def _clip_detections(image, detections):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds  = ((detections[:, 2] - detections[:, 0]) > 0) & \
                 ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections

def cornernet(system_configs, db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size

    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
    output_size  = db.configs["output_sizes"][0]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]

    max_tag_len = 1024

    # allocating memory
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
    tl_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    br_regrs    = np.zeros((batch_size, max_tag_len, 2), dtype=np.float32)
    tl_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    br_tags     = np.zeros((batch_size, max_tag_len), dtype=np.int64)
    tag_masks   = np.zeros((batch_size, max_tag_len), dtype=np.uint8)
    tag_lens    = np.zeros((batch_size, ), dtype=np.int32)

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_path = db.image_path(db_ind)
        # print(image_path)
        # image      = cv2.imread(image_path)
        img = np.load(image_path).astype(np.uint8)
        pre = img[:, :, 0:3]
        cur = img[:, :, 3:6]
        post = img[:, :, 6:9]

        # reading detections
        # print(db_ind)
        detections = db.detections(db_ind)

        # cropping an image randomly
        if not debug and rand_crop:
            # image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
            cur, detections_cur = random_crop(cur, detections, rand_scales, input_size, border=border)
            pre, detections_pre = random_crop(pre, detections, rand_scales, input_size, border=border)
            post, detections_post = random_crop(post, detections, rand_scales, input_size, border=border)

        # image, detections = _resize_image(image, detections, input_size)
        # detections = _clip_detections(image, detections)
        cur, detections_cur = _resize_image(cur, detections_cur, input_size)
        detections_cur = _clip_detections(cur, detections_cur)

        pre, detections_pre = _resize_image(pre, detections_pre, input_size)
        detections_pre = _clip_detections(pre, detections_pre)

        post, detections_post = _resize_image(post, detections_post, input_size)
        detections_post = _clip_detections(post, detections_post)

        width_ratio  = output_size[1] / input_size[1]
        height_ratio = output_size[0] / input_size[0]

        # flipping an image randomly
        if not debug and np.random.uniform() > 0.5:
            # image[:] = image[:, ::-1, :]
            cur[:] = cur[:, ::-1, :]
            pre[:] = pre[:, ::-1, :]
            post[:] = post[:, ::-1, :]

            width    = cur.shape[1]
            detections_cur[:, [0, 2]] = width - detections_cur[:, [2, 0]] - 1

        if not debug:
            # image = image.astype(np.float32) / 255.
            cur = cur.astype(np.float32) / 255.
            pre = pre.astype(np.float32) / 255.
            post = post.astype(np.float32) / 255.
            if rand_color:
                # color_jittering_(data_rng, image)
                color_jittering_(data_rng, cur)
                color_jittering_(data_rng, pre)
                color_jittering_(data_rng, post)
                if lighting:
                    # lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
                    lighting_(data_rng, cur, 0.1, db.eig_val, db.eig_vec)
                    lighting_(data_rng, pre, 0.1, db.eig_val, db.eig_vec)
                    lighting_(data_rng, post, 0.1, db.eig_val, db.eig_vec)
            # normalize_(image, db.mean, db.std)
            normalize_(cur, db.mean, db.std)
            normalize_(pre, db.mean, db.std)
            normalize_(post, db.mean, db.std)
        # image = np.dstack((pre, cur, post))
        image = cur
        images[b_ind] = image.transpose((2, 0, 1))
        detections = detections_cur

        for ind, detection in enumerate(detections):
            print(ind)
            # print(detection)
            category = int(detection[-1]) - 1
            # print(category)

            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            if gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)

                if gaussian_rad == -1:
                    radius = gaussian_radius((height, width), gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = gaussian_rad

                draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
                draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
            else:
                tl_heatmaps[b_ind, category, ytl, xtl] = 1
                br_heatmaps[b_ind, category, ybr, xbr] = 1

            tag_ind = tag_lens[b_ind]
            tl_regrs[b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
            br_regrs[b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
            tl_tags[b_ind, tag_ind] = ytl * output_size[1] + xtl
            br_tags[b_ind, tag_ind] = ybr * output_size[1] + xbr
            tag_lens[b_ind] += 1

    for b_ind in range(batch_size):
        tag_len = tag_lens[b_ind]
        tag_masks[b_ind, :tag_len] = 1

    images      = torch.from_numpy(images)
    tl_heatmaps = torch.from_numpy(tl_heatmaps)
    br_heatmaps = torch.from_numpy(br_heatmaps)
    tl_regrs    = torch.from_numpy(tl_regrs)
    br_regrs    = torch.from_numpy(br_regrs)
    tl_tags     = torch.from_numpy(tl_tags)
    br_tags     = torch.from_numpy(br_tags)
    tag_masks   = torch.from_numpy(tag_masks)

    return {
        "xs": [images],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, tl_tags, br_tags]
    }, k_ind
