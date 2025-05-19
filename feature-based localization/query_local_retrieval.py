import os
import copy
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from lightglue.utils import load_image, rbd
from lightglue import LightGlue, SuperPoint, viz2d


if __name__ == "__main__":
    matching_rst_fd = 'data\\matching_rst'
    GSV_meta_fd = 'E:\\VPN\\RichmondCollect_filtered\\GSV\\GSV_meta'
    query_fds = [os.path.join(matching_rst_fd, item) for item in os.listdir(matching_rst_fd)]
    transformer = Transformer.from_crs(4326, 6347)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mkpt = 1024
    extractor = SuperPoint(max_num_keypoints=mkpt).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    for fd in tqdm(query_fds, desc=f'Local matching score computing'):
        matching_rst = pd.read_csv(os.path.join(fd, 'global_matching.csv'), index_col='GSV_label')
        query_fp = os.path.join(fd, 'template.jpg')

        img_q = load_image(query_fp)
        feats_q = extractor.extract(img_q.to(device))

        GSV_fps = [os.path.join(fd, item) for item in os.listdir(fd) if 'SV_' in item]
        xys, lbls = [], []

        for fp in GSV_fps:
            name = os.path.basename(fp)
            lbls.append(name.split('.')[0])
            GSV_meta_fp = os.path.join(GSV_meta_fd, name.replace('.jpg', '.metadata.json'))
            info = json.loads(open(GSV_meta_fp, 'r').read())
            xy = transformer.transform(info['lat'], info['lng'])
            xys.append([xy[0], xy[1], info['elevation']])
        xys = np.array(xys)

        clustering = DBSCAN(eps=50, min_samples=1).fit(xys[:, 0:2])
        unique_lbl = np.unique(clustering.labels_)
        unique_lbl = np.delete(unique_lbl, np.where(unique_lbl == -1))
        common_num, GSVs, local_score = [], [], []

        for lbl in unique_lbl:
            GSV_names, feats_ids = [], []
            ids = np.where(clustering.labels_ == lbl)[0]
            for idx in ids:
                GSV_names.append(os.path.basename(GSV_fps[idx]))

            for GSV_name in GSV_names:
                img_g_fp = os.path.join(fd, GSV_name)
                img_g = load_image(img_g_fp)
                feats_g = extractor.extract(img_g.to(device))
                matches_qg = matcher({"image0": feats_q, "image1": feats_g})
                feats_q_, feats_g_, matches_qg = [
                    rbd(x) for x in [feats_q, feats_g, matches_qg]
                ]
                matches = matches_qg["matches"]
                feats_ids.append(set(matches[..., 0].data.detach().cpu().numpy()))

                # if 'Commons_Luma_Rectified_Left_270' in fd:
                if 'Commons_Luma_Rectified_Left_270' in fd:
                    kptsq, kptsg = feats_q_["keypoints"], feats_g_["keypoints"]
                    m_kptsq, m_kptsg = kptsq[matches[..., 0]], kptsg[matches[..., 1]]
                    c = 1
                    # axes = viz2d.plot_images([img_q, img_g])
                    # viz2d.plot_matches(m_kptsq, m_kptsg, color="lime", lw=0.2)
                    # viz2d.add_text(0, f'Stop after {matches_qg["stop"]} layers')
                    #
                    # kpc0, kpc1 = viz2d.cm_prune(matches_qg["prune0"]), viz2d.cm_prune(matches_qg["prune1"])
                    # viz2d.plot_images([img_q, img_g])
                    # viz2d.plot_keypoints([kptsq, kptsg], colors=[kpc0, kpc1], ps=6)

            if len(feats_ids) == 1:
                common_num.append(len(feats_ids[0]))
            else:
                common_ids = copy.deepcopy(feats_ids[0])
                for idx in range(1, len(feats_ids)):
                    common_ids = common_ids.union(feats_ids[idx])
                common_num.append(len(common_ids))

            GSVs.append(','.join(GSV_names))
            local_score.append(common_num[-1] / mkpt)
        rst_df = pd.DataFrame({'GSV_label': GSVs,
                               'common': common_num,
                               'local_score': local_score})
        out_fp = os.path.join(fd, 'local_matching.csv')
        if os.path.exists(out_fp):
            os.remove(out_fp)
        rst_df.to_csv(out_fp, index=False)
