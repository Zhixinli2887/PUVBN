import os
import json
import osmnx as ox


def bearing_to_vec(bearing):
    c = 1


if __name__ == "__main__":
    GSV_img_fd = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\GSV\\IMG\\1819'
    GSV_info_fd = 'E:\\VPN\\proj_file\\PUVBN\\PU_GSV\\GSV\\GSV_meta\\1819'

    G = ox.graph_from_place('West Lafayette, IN, USA', network_type='drive')
    Gu = ox.add_edge_bearings(ox.get_undirected(G))
    roads = []

    for u, v, k, data in Gu.edges(keys=True, data=True):
        c = 1

    GSV_names, GSV_xyz = [], []
    for fn in os.listdir(GSV_info_fd):
        fp = os.path.join(GSV_info_fd, fn)
        item = json.loads(open(fp, 'r').read())
        GSV_names.append(item['filename'])
        lat, lng = item['lat'], item['lng']

        G = ox.graph_from_place('West Lafayette, IN, USA', network_type='drive')
        Gu = ox.add_edge_bearings(ox.get_undirected(G))

        edge = ox.nearest_edges(G, X=[lat], Y=[lng], return_dist=True)
        c = 1
