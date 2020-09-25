from tqdm import tqdm
import torch.nn as nn

from utils.config import *
from models.CDST import *
from utils.process_data import *

'''
python train.py -dec= -bsz= -hdd= -dr= -lr=
'''

early_stop = args['earlyStop']


def main():
    print("Running training process")
    # Configure models and load data
    avg_best, cnt, acc = 0.0, 0, 0.0
    train, dev, test, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, False,
                                                                                 batch_size=int(args['batch']))

    model = globals()[args['decoder']](
        hidden_size=int(args['hidden']),
        lang=lang,
        path=args['path'],
        task=args['task'],
        lr=float(args['lr_rate']),
        dropout=float(args['dropout']),
        slots=SLOTS_LIST,
        gating_dict=gating_dict,
        nb_train_vocab=max_word)

    for epoch in range(500):
        print("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:
            model.train_batch(data, int(args['clip']), SLOTS_LIST[1], SLOTS_LIST[2], reset=(i == 0))
            model.optimize(args['clip'])
            pbar.set_description(model.print_loss())

        if (epoch + 1) % int(args['evalp']) == 0:

            acc = model.evaluate(dev, avg_best, SLOTS_LIST[3], SLOTS_LIST[4], early_stop)
            model.scheduler.step(acc)

            if acc >= avg_best:
                avg_best = acc
                cnt = 0
                best_model = model
            else:
                cnt += 1

            if cnt == args["patience"] or (acc == 1.0 and early_stop is None):
                print("Ran out of patient, early stop...")
                break


if __name__ == '__main__':
    main()
