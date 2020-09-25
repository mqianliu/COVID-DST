import random
import json


def get_data_spilt():
    train_num = 480
    dev_num = 60
    test_num = 60
    dev_end = train_num + dev_num
    test_end = dev_end + test_num
    data_spilt = {}

    train_idx = [0]
    dev_idx = [1]
    test_idx = [3]

    idxs = [i for i in range(0, 603)]
    idxs.remove(0)
    idxs.remove(1)
    idxs.remove(3)
    random.shuffle(idxs)
    train_idx = train_idx + idxs[:train_num]
    dev_idx = dev_idx + idxs[train_num:dev_end]
    test_idx = test_idx + idxs[dev_end:test_end]
    data_spilt["train"] = train_idx
    data_spilt["dev"] = dev_idx
    data_spilt["test"] = test_idx
    # with open('data/data_split.json', 'w') as f:
    #     json.dump(data_spilt, f)


# if __name__ == '__main__':
    # get_data_spilt()