def get(dataset_path, dataset_name):
    feat_dataset_file = ""
    if dataset_name == "products":
        path =  dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif dataset_name == "paper100m":
        path = dataset_path + "/paper100m/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 1207179  
        valid_set_num = 125265
        test_set_num = 214338
    elif dataset_name == "com-friendster":
        path = dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif dataset_name == "ukunion":
        path = dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 116
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif dataset_name == "uk2014":
        path = dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif dataset_name == "clueweb":
        path = dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    elif dataset_name == "cora":
        path = dataset_path + "/cora/"
        vertices_num = 19793
        edges_num = 126842
        features_dim = 8710
        train_set_num = 7917
        valid_set_num = 1979
        test_set_num = 9897
    elif dataset_name == "reddit":
        path = dataset_path + "/reddit/"
        vertices_num = 232965
        edges_num = 114615892
        features_dim = 602
        train_set_num = 153431
        valid_set_num = 23831
        test_set_num = 55703
    elif dataset_name == "pubmed":
        path = dataset_path + "/pubmed/"
        vertices_num = 19717
        edges_num = 88651
        features_dim = 500
        train_set_num = 7886
        valid_set_num = 1972
        test_set_num = 9859
    elif dataset_name == "mag":
        path = dataset_path + "/mag/"
        vertices_num = 121751666
        edges_num = 1297748926
        features_dim = 384
        train_set_num = 1112392
        valid_set_num = 138949
        test_set_num = 88092
    elif dataset_name == "citeseer":
        path = dataset_path + "/citeseer/"
        vertices_num = 3327
        edges_num = 9228
        features_dim = 3703
        train_set_num = 1330
        valid_set_num = 333
        test_set_num = 1664
    elif dataset_name == "pubmed_ls":
        path = dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 500
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
        feat_dataset_file = dataset_path + "/pubmed/"
    elif dataset_name == "citeseer_ls":
        path =  dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 3703
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
        feat_dataset_file = dataset_path + "/citeseer/"
    elif dataset_name == "cora_ls":
        path =  dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 8710
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
        feat_dataset_file = dataset_path + "/cora/"
    else:
        print("invalid dataset path")
        exit

    return path, vertices_num, edges_num, features_dim, train_set_num, \
            valid_set_num, test_set_num, feat_dataset_file