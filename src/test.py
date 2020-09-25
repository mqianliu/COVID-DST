from utils.config import *
from models.CDST import *
from utils.process_data import *

'''
python3 myTest.py -ds= -path= -bsz=
'''


def main():
    directory = args['path'].split("/")
    HDD = directory[2].split('HDD')[1].split('BSZ')[0]
    decoder = directory[1].split('-')[0]
    BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
    args["decoder"] = decoder
    args["HDD"] = HDD
    print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)

    train, dev, test, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, False,
                                                                                 batch_size=int(args['batch']))

    model = globals()[decoder](
        int(HDD),
        lang=lang,
        path=args['path'],
        task=args["task"],
        lr=0,
        dropout=0,
        slots=SLOTS_LIST,
        gating_dict=gating_dict,
        nb_train_vocab=max_word)

    if args["run_dev_testing"]:
        print("Development Set ...")
        acc_dev = model.evaluate(dev, 1e7, SLOTS_LIST[3], SLOTS_LIST[4])

    print("Test Set ...")
    acc_test = model.evaluate(test, 1e7, SLOTS_LIST[5], SLOTS_LIST[6])


if __name__ == '__main__':
    main()
