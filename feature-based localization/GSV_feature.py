import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from py360convert import e2p
import matplotlib.pyplot as plt
from multiprocessing import RLock
from transformers import CLIPModel, AutoProcessor

model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_img_feat(image):
    inputs = processor(images=image, return_tensors="pt")
    feat = model_CLIP.get_image_features(**inputs)
    return feat.cpu().detach().numpy()[0]


def f(fp):
    pid = mp.current_process()._identity[0] - 1
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    out_fp = os.path.join('data\\GSV_feat', fp.replace('.jpg', '.npy'))

    if not os.path.exists(out_fp):
        FOV = (74, 50)
        u_degs = [5 * item - 180 for item in range(int(360 / 5))]
        v_degs = [5 * item - 10 for item in range(6)]
        img = cv2.imread(os.path.join(GSV_fd, fp))
        out_img_size = (int(1200 / 4), int(1920 / 4))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray

        img_feats, uvs = [], []
        for u_deg in u_degs:
            for v_deg in v_degs:
                uvs.append([u_deg, v_deg])

        for uv in tqdm(uvs, desc=f'Processing {fp}', position=pid):
            img_prj = e2p(img, FOV, uv[0], uv[1], out_img_size)
            img_feat = get_img_feat(img_prj)
            img_feats.append(img_feat)
        uvs = np.array(uvs)
        img_feats = np.array(img_feats)
        np.save(out_fp, np.concatenate([uvs, img_feats], axis=1))
    return True


if __name__ == "__main__":
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'
    gray_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG_gray'

    fps = [item for item in os.listdir(GSV_fd) if item in os.listdir(gray_fd)]
    p = mp.Pool(2, initargs=(RLock(),), initializer=tqdm.set_lock)
    jobs = [p.apply_async(f, args=[fp]) for fp in fps]
    result_list = [job.get() for job in jobs]
    p.close()
