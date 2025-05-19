import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from py360convert import e2p, e2c


def project(fp):
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    out_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\projected'
    out_gray_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG_gray'

    w_deg, h_deg = 90, 120
    FOV = (w_deg, h_deg)
    df = 12
    pix_unit = int(36 / df)
    out_img_size = (h_deg * pix_unit, w_deg * pix_unit)
    degs = [w_deg * item - 180 for item in range(int(540 / w_deg))]
    img = cv2.imread(os.path.join(GSV_fd, fp))

    sh, sw, _ = img.shape
    # down_size = (int(sw / df), int(sh / df))
    down_size = (1110, 555)
    image_raw = cv2.resize(img, down_size)
    gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image_raw[:, :, 0] = gray
    image_raw[:, :, 1] = gray
    image_raw[:, :, 2] = gray
    cv2.imwrite(os.path.join(out_gray_fd, fp), image_raw)

    out_fp = os.path.join(out_fd, fp)
    w = w_deg * pix_unit
    # img_rgb = e2c(img, face_w=w, mode='bilinear', cube_format='dice')
    # img_rgb = img_rgb[w:(2 * w), :]
    # img_rgb = np.concatenate((img_rgb, img_rgb[:, 0:int(2 * w), :]), axis=1)
    GSV_prjs = []

    for u_deg in degs:
        GSV_prjs.append(e2p(img, FOV, u_deg, 0, out_img_size))

    img_rgb = np.concatenate(GSV_prjs, axis=1)
    img_rgb = np.concatenate((img_rgb[:, int(w / 2):, :], img_rgb[:, int(w * 2):int(w * 2 + w / 2), :]), axis=1)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out_fp, img_gray)
    return True


if __name__ == "__main__":
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    gray_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG_gray'

    fps = [item for item in os.listdir(GSV_fd) if item in os.listdir(gray_fd)]

    p = mp.Pool(6)
    r = list(tqdm(p.imap(project, fps), total=len(fps), desc='Projecting...'))
