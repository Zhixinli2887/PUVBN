import cv2
import mmcv
import shutil
import os.path
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from py360convert import e2p
import matplotlib.pyplot as plt
from transformers import CLIPModel, AutoProcessor
from sklearn.metrics.pairwise import cosine_similarity
from mmseg.apis import inference_model, init_model, show_result_pyplot


config_file = 'data/segmodel/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py'
checkpoint_file = 'data/segmodel/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'
model_seg = init_model(config_file, checkpoint_file, device='cuda:0')
model_CLIP = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
warnings.filterwarnings("ignore")


def get_img_feat(image):
    inputs = processor(images=image, return_tensors="pt")
    feat = model_CLIP.get_image_features(**inputs)
    return feat.cpu().detach().numpy()[0]


if __name__ == "__main__":
    temp_query = 'temp_query'
    GSV_feat_fd = 'data\\GSV_feat'
    GSV_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\IMG'

    GSV_feat_fps = [os.path.join(GSV_feat_fd, item) for item in os.listdir(GSV_feat_fd)]
    GSV_feats, lbls, uvs, idx = [], [], [], None

    for fp in tqdm(GSV_feat_fps, desc='Loading GSV global features'):
        lbls.append(os.path.basename(fp).split('.')[0])
        feat = np.load(fp)
        if len(uvs) == 0:
            idx = np.where(feat[:, 1] <= 15)[0]
            uvs = feat[idx, 0:2]
        feat = feat[idx, 2:]
        GSV_feats.append(feat)
    GSV_feats = np.array(GSV_feats)

    k, feat_num, img_size = 10, len(GSV_feats), (1200, 1920)
    # down_size = (int(1200 / 4), int(1920 / 4))
    p_labels = [2, 3, 4, 5, 6, 7]
    FOV = (74, 50)

    matching_rst_fd = 'data\\matching_rst'
    if os.path.exists(matching_rst_fd):
        shutil.rmtree(matching_rst_fd)
    os.mkdir(matching_rst_fd)

    sensors = ['s27']

    for sensor in sensors:
        query_fd = f'data\\query\\{sensor}'
        fps = [item for item in os.listdir(query_fd) if 'Rectified' in item and 'new' not in item]

        for query_fp in fps:
            print(f'Working on query image {Path(query_fp).stem}:..')

            if os.path.exists(temp_query):
                shutil.rmtree(temp_query)
            os.mkdir(temp_query)

            query_fp = os.path.join(query_fd, query_fp)
            shutil.copy(query_fp, os.path.join(temp_query, os.path.basename(query_fp)))
            query_fp = os.path.join(temp_query, os.path.basename(query_fp))

            seg_result = inference_model(model_seg, query_fp)
            # seg_img = show_result_pyplot(model_seg, query_fp, seg_result, with_labels=False, opacity=.8)
            query_label = seg_result.get('pred_sem_seg').data.detach().cpu().numpy()[0]
            seg_mask = np.isin(query_label, p_labels)
            query_p = np.sum(seg_mask) / (seg_mask.shape[0] * seg_mask.shape[1])

            if query_p < 0.1:
                print(f'No sufficient permanent objects, continue to next..')
            else:
                query_feat = get_img_feat(cv2.imread(query_fp))
                uv_ids, scores = [], []

                temp_rst_fd = matching_rst_fd + f'\\{Path(query_fp).stem}'
                if os.path.exists(temp_rst_fd):
                    shutil.rmtree(temp_rst_fd)
                os.mkdir(temp_rst_fd)

                shutil.copy(query_fp, os.path.join(temp_rst_fd, 'template.jpg'))
                us, vs = [], []

                for idx in tqdm(range(feat_num), desc='Computing similarity'):
                    cosine_score = cosine_similarity([query_feat], GSV_feats[idx])
                    uv_ids.append(np.argmax(cosine_score))
                    scores.append(np.max(cosine_score))
                    uv = uvs[np.argmax(cosine_score)]
                    us.append(uv[0])
                    vs.append(uv[1])

                    # if 'Monroe_Luma_Rectified_Right_330' in temp_rst_fd:
                    #     if 'SV_709' in lbls[idx]:
                    #         import matplotlib as mpl
                    #         cmap = mpl.colormaps['RdYlGn']
                    #         colors = cmap(np.linspace(0, 1, cosine_score.shape[1]))
                    #         GSV_img = cv2.imread(os.path.join(GSV_fd, lbls[idx] + '.jpg'), cv2.IMREAD_GRAYSCALE)
                    #         plt.imshow(GSV_img, cmap='gray', vmin=0, vmax=255)
                    #         mins, maxs = cosine_score.min(), cosine_score.max()
                    #         for i in range(len(uvs)):
                    #             u, v = uvs[i, 0], uvs[i, 1]
                    #             x = int((u + 180) / 360 * GSV_img.shape[1])
                    #             y = int((v + 90) / 180 * GSV_img.shape[0])
                    #             plt.scatter(x, y, 150 * (cosine_score[0][i] - mins) / (maxs - mins),
                    #                         color='red')
                    #         plt.show()

                top_ids = np.argsort(scores)[::-1][0:k]
                for idx in tqdm(top_ids, desc='Projecting'):
                    GSV_name = lbls[idx] + '.jpg'
                    GSV_img = cv2.imread(os.path.join(GSV_fd, GSV_name))
                    uv = uvs[uv_ids[idx]]
                    GSV_patch = e2p(GSV_img, FOV, uv[0], uv[1], img_size)
                    cv2.imwrite(os.path.join(temp_rst_fd, GSV_name), GSV_patch)

                rst_fp = os.path.join(temp_rst_fd, 'global_matching.csv')
                rst_df = pd.DataFrame({'GSV_label': lbls,
                                       'matching_score': scores,
                                       'u_deg': us,
                                       'v_deg': vs})
                rst_df.to_csv(rst_fp, index=False)
