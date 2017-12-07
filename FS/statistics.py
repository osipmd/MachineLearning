import sys

from analyse import do_analyse


def create_statistics():
    pearson_sorted_keys_all, spearman_sorted_keys_all, IG_sorted_keys_all = do_analyse()
    pearson_sorted_keys = pearson_sorted_keys_all[:20]
    spearman_sorted_keys = spearman_sorted_keys_all[:20]
    IG_sorted_keys = IG_sorted_keys_all[:20]

    pearson = []
    pearson_spearman = []
    pearson_IG = []
    spearman = []
    spearman_IG = []
    IG = []
    All = []
    for i in range(len(pearson_sorted_keys)):
        x = pearson_sorted_keys[i]
        if x in spearman_sorted_keys and x in IG_sorted_keys:
            All.append(x)
        elif x in spearman_sorted_keys:
            pearson_spearman.append(x)
        elif x in IG_sorted_keys:
            pearson_IG.append(x)
        else:
            pearson.append(x)
    for i in range(len(spearman_sorted_keys)):
        x = spearman_sorted_keys[i]
        if x in pearson_sorted_keys and x in IG_sorted_keys and x not in All:
            All.append(x)
        elif x in pearson_sorted_keys and x not in pearson_spearman:
            pearson_spearman.append(x)
        elif x in IG_sorted_keys:
            spearman_IG.append(x)
        else:
            spearman.append(x)
    for i in range(len(IG_sorted_keys)):
        x = IG_sorted_keys[i]
        if x in pearson_sorted_keys and x in spearman_sorted_keys and x not in All:
            All.append(x)
        elif x in pearson_sorted_keys and x not in pearson_IG:
            pearson_IG.append(x)
        elif x in spearman_sorted_keys and x not in spearman_IG:
            spearman_IG.append(x)
        else:
            IG.append(x)
        #    print("All : ", All)
        #    print('pearson_spearman : ', pearson_spearman)
        #    print('pearson_IG : ', pearson_IG)
        #    print('spearman_IG : ', spearman_IG)
        #    print('pearson : ', pearson)
        #    print('spearman : ', spearman)
        #    print('IG : ', IG)

    default = sys.stdout
    with open('results.txt', 'w') as out:
        sys.stdout = out
        print("psk", ":", pearson_sorted_keys_all)
        print("ssk", ":", spearman_sorted_keys_all)
        print("isk", ":", IG_sorted_keys_all)

        print("All : ", All)
        print('pearson_spearman : ', pearson_spearman)
        print('pearson_IG : ', pearson_IG)
        print('spearman_IG : ', spearman_IG)
        print('pearson : ', pearson)
        print('spearman : ', spearman)
        print('IG : ', IG)
    sys.stdout = default
    return All, pearson_spearman, pearson_IG, spearman_IG, pearson, spearman, IG


create_statistics()