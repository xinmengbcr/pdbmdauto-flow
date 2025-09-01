import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class GroAtom:
    def __init__(self, resid, residuename, atomname, index, x, y, z):
        self.resid = resid
        self.residuename = residuename
        self.atomname = atomname
        self.index = index
        self.x = x
        self.y = y
        self.z = z


def loadGro_removePBC(inputfile, outfile, inresidue_thre=1, inchain_thre=3):
    _REMARK = 2
    with open(inputfile) as f:
        lines = f.readlines()
        num_atoms = int(lines[1])
        box_size = np.array(
            lines[_REMARK + num_atoms].strip("\n").lstrip().split("  "), dtype=float
        )
    coord_array = []
    resid_list = []
    residuename_list = []
    atomname_list = []
    index_list = []
    x_list = []
    y_list = []
    z_list = []

    with open(inputfile, "r") as f:
        data = f.readlines()
        count = 0
        for line in data[_REMARK : _REMARK + num_atoms]:
            count += 1
            # the resid and residuename belong to list[0],split it
            resid = int(line[0:5])  # int( (count-1) / 121 ) +1
            residuename = line[5:10]
            atomname = line[10:15]
            index = int(line[15:20])
            # double in c, float64 in numpy --> doesn't matter in here
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])

            resid_list.append(resid)
            residuename_list.append(residuename)
            atomname_list.append(atomname)
            index_list.append(index)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

    df = pd.DataFrame(
        {
            "resid": resid_list,
            "residuename": residuename_list,
            "atomname": atomname_list,
            "atomid": index_list,
            "x": x_list,
            "y": y_list,
            "z": z_list,
        }
    )
    # df.to_csv('test.csv',index=False)

    resid_list = set(df["resid"])
    # print(resid_list)

    for resid in resid_list:
        group = df[(df["resid"] == resid)]
        coord_x_list = group["x"]
        coord_y_list = group["y"]
        coord_z_list = group["z"]
        coord_array = np.array([coord_x_list, coord_y_list, coord_z_list]).T
        clustering = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=inresidue_thre
        ).fit(coord_array)
        # print(resid, clustering.labels_)
        # print(np.unique( clustering.labels_))
        if len(np.unique(clustering.labels_)) > 1:
            # print(resid, clustering.labels_)
            _df = pd.DataFrame(
                {
                    "label": clustering.labels_,
                    "x": coord_x_list,
                    "y": coord_y_list,
                    "z": coord_z_list,
                }
            )
            # print(_df)
            # Here use the 0 group as reference; and assume no -1 group label
            # should be only two groups ...
            group_ref = _df.copy()[(_df["label"] == 0)]
            com_x_group_ref = group_ref["x"].mean()
            com_y_group_ref = group_ref["y"].mean()
            com_z_group_ref = group_ref["z"].mean()

            # print(ref_group)
            labels = np.unique(_df.label)
            # print(labels)
            check_lables = np.delete(labels, np.where(labels == 0))
            for label in check_lables:
                # print(label)
                group_check = _df.copy()[(_df["label"] == label)]
                com_x_group_check = group_check["x"].mean()
                com_y_group_check = group_check["y"].mean()
                com_z_group_check = group_check["z"].mean()

                # Thanks pandas, the index from df is kept in the following operations
                indices = (group_check.index).tolist()
                # can use https://stackoverflow.com/questions/37725195/pandas-replace-values-based-on-index
                # print(indices)
                # print(group_check)
                if (com_x_group_check - com_x_group_ref) >= box_size[0] / 2.0:
                    list_coord_demo = group_check["x"] - box_size[0]
                    df.loc[indices, "x"] = list_coord_demo
                elif (com_x_group_check - com_x_group_ref) <= -box_size[0] / 2.0:
                    list_coord_demo = group_check["x"] + box_size[0]
                    df.loc[indices, "x"] = list_coord_demo
                else:
                    pass

                if (com_y_group_check - com_y_group_ref) >= box_size[1] / 2.0:
                    list_coord_demo = group_check["y"] - box_size[1]
                    df.loc[indices, "y"] = list_coord_demo
                elif (com_y_group_check - com_y_group_ref) <= -box_size[1] / 2.0:
                    list_coord_demo = group_check["y"] + box_size[1]
                    df.loc[indices, "y"] = list_coord_demo
                else:
                    pass

                if (com_z_group_check - com_z_group_ref) >= box_size[2] / 2.0:
                    list_coord_demo = group_check["z"] - box_size[2]
                    df.loc[indices, "z"] = list_coord_demo
                elif (com_z_group_check - com_z_group_ref) <= -box_size[2] / 2.0:
                    list_coord_demo = group_check["z"] + box_size[2]
                    df.loc[indices, "z"] = list_coord_demo
                else:
                    pass

    # to see the residue is full..
    df2 = df.copy()
    resid_list = set(df2["resid"])
    # print(resid_list)
    for resid in resid_list:
        group = df2[(df2["resid"] == resid)]
        coord_x_list = group["x"]
        coord_y_list = group["y"]
        coord_z_list = group["z"]
        coord_array = np.array([coord_x_list, coord_y_list, coord_z_list]).T
        clustering = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=inresidue_thre
        ).fit(coord_array)
        # print(resid, clustering.labels_)
        # print(np.unique( clustering.labels_))
        if len(np.unique(clustering.labels_)) > 1:
            print("ERROR!!!! residue not full!!!!! ")
    # residue is full

    # To make the chain full
    df2 = df.copy()
    list_resid = np.unique(df2["resid"]).tolist()
    # print(list_resid) This is check the center of mass
    list_com_residue = []  # vs residue_com_list  naming style
    for resid in list_resid:
        group = df2[(df2["resid"] == resid)]
        coord_x_list = group["x"]
        coord_y_list = group["y"]
        coord_z_list = group["z"]
        com_x = group["x"].mean()
        com_y = group["y"].mean()
        com_z = group["z"].mean()
        list_com_residue.append([com_x, com_y, com_z])
    # not based on reisd
    # coord_x_list = df2["x"]
    # coord_y_list = df2["y"]
    # coord_z_list = df2["z"]
    # coord_array = np.array([ coord_x_list, coord_y_list, coord_z_list]).T
    # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=20).fit(coord_array)
    # print(clustering.labels_)

    # sort residue cluster based on the distance inchain_thre
    array_com = np.array(list_com_residue)
    clustering_chain = AgglomerativeClustering(
        n_clusters=None, linkage="single", distance_threshold=inchain_thre
    ).fit(array_com)
    # print(clustering_chain.labels_)
    # print(list_resid)
    # print(np.unique( clustering_chain.labels_))
    # start to move residues together

    _df2 = pd.DataFrame({"label": clustering_chain.labels_, "resid": list_resid})
    # print(_df2)
    labels = np.unique(_df2.label)
    # print(labels)

    # here group is the whole resid
    # ref select as the largest cluster
    counts = np.bincount(clustering_chain.labels_)
    select_ref = np.argmax(counts)
    # print('select reference cluster:  ', select_ref)
    # get all the resids in the ref cluster
    list_resid_ref = _df2.copy()[(_df2["label"] == select_ref)]["resid"].tolist()
    # print(list_resid_ref)

    # ref group ---> all the atoms in the ref group resids !!
    group_ref = df2.copy()[df2["resid"].isin(list_resid_ref)]
    coord_x_list = group_ref["x"]
    coord_y_list = group_ref["y"]
    coord_z_list = group_ref["z"]
    array_ref = np.array([coord_x_list, coord_y_list, coord_z_list]).T
    # demo check the ref is in same resid
    # clustering_demo = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=inchain_thre).fit(   array_ref  )
    # print(clustering_demo.labels_)
    # _result = np.unique(clustering_demo.labels_)
    # print(_result)

    check_lables = np.delete(labels, np.where(labels == select_ref))
    for label in check_lables:
        # print(label)
        list_resid_check = _df2.copy()[(_df2["label"] == label)]["resid"].tolist()
        # print(list_resid_check )
        ###
        # https://stackoverflow.com/questions/12096252/use-a-list-of-values-to-select-rows-from-a-pandas-dataframe
        group_check = df2.copy()[df2["resid"].isin(list_resid_check)]
        indices = (group_check.index).tolist()
        # print(indices)
        coord_x_list = group_check["x"]
        coord_y_list = group_check["y"]
        coord_z_list = group_check["z"]
        array_check = np.array([coord_x_list, coord_y_list, coord_z_list]).T
        # check clustering
        # merge array np.concatenate((a, b), axis=0) #https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
        clustering_ori = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=inchain_thre
        ).fit(np.concatenate((array_ref, array_check), axis=0))
        # print(clustering_demo.labels_)
        length_ori = len(np.unique(clustering_ori.labels_))
        # print(label, length_ori)

        _list = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([1, 1, 0]),
            np.array([1, 0, 1]),
            np.array([0, 1, 1]),
            np.array([1, 1, 1]),
        ] + [
            -np.array([1, 0, 0]),
            -np.array([0, 1, 0]),
            -np.array([0, 0, 1]),
            -np.array([1, 1, 0]),
            -np.array([1, 0, 1]),
            -np.array([0, 1, 1]),
            -np.array([1, 1, 1]),
        ]

        # print(_list)
        status = "N"
        for operation in _list:
            _array = array_check + operation * box_size
            clustering_demo = AgglomerativeClustering(
                n_clusters=None, linkage="single", distance_threshold=inchain_thre
            ).fit(np.concatenate((array_ref, _array), axis=0))
            length_aftermov = len(np.unique(clustering_demo.labels_))
            if length_aftermov == 1:
                # print('done',operation)
                df2.loc[indices, "x"] = _array[:, 0]
                df2.loc[indices, "y"] = _array[:, 1]
                df2.loc[indices, "z"] = _array[:, 2]
                status = "Y"
                break
                # small inchain_thre may be problematic if the reisdue move to approiate but still break with neightbur gap (not moved yet)
                # !!!!!!!!!  problematic for the following ....
                # !!!!!!!!!                    ---         ---        ----
                # !!!!!!!!!  -------   >thre        >thre      >thre         >thre   ----------
                # Could use recursive; do it again and agian until yes ....
                # Now here just use a relative large inchain_thre
                # or increase thre...
        # print(label, status)

    # df2.to_csv('test_fix2.csv',index=False)

    with open(outfile, "w") as fo:
        fo.write("MOLECULE" + "\n")
        fo.write("%5d " % (num_atoms) + "\n")  # n1*n2  n1*n2*2
        # for i in xrange( num_atoms ):
        for index, row in df2.iterrows():
            # added 18-02-2016
            if row.resid > 99999:
                row.resid = row.resid % 99999
            if row.atomid > 99999:
                row.atomid = row.atomid % 99999
            fo.write(
                "%5d%5s%5s%5d%8.3f%8.3f%8.3f"
                % (
                    row.resid,
                    row.residuename,
                    row.atomname,
                    row.atomid,
                    row.x,
                    row.y,
                    row.z,
                )
                + "\n"
            )
        fo.write("%10.6f%10.6f%10.6f" % (box_size[0], box_size[1], box_size[2]) + "\n")
        fo.write("\n")

    coord_x_list = df2["x"]
    coord_y_list = df2["y"]
    coord_z_list = df2["z"]
    array_ref = np.array([coord_x_list, coord_y_list, coord_z_list]).T
    clustering_demo = AgglomerativeClustering(
        n_clusters=None, linkage="single", distance_threshold=inchain_thre
    ).fit(array_ref)
    # print(clustering_demo.labels_)
    status = len(np.unique(clustering_demo.labels_))
    if status == 1:
        result = {"success": True, "message": "success"}
    else:
        result = {"success": False, "message": "fail"}
    return json.dumps(result)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            result = {"success": False, "error": "No parameters provided"}
            print(json.dumps(result), flush=True)
            sys.exit(1)

        params = json.loads(sys.argv[1])
        if not os.path.exists(params["input_file"]):
            result = {
                "success": False,
                "error": f"Input file not found: {params['input_file']}",
            }
            print(json.dumps(result), flush=True)
            sys.exit(1)

        result = loadGro_removePBC(params["input_file"], params["output_file"])
        print(result, flush=True)  # result is already JSON string

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_result), flush=True)
        sys.exit(1)
