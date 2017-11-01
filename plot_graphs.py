"""
Given a logfile, plot a graph
"""
import matplotlib.pyplot as plt
import json
import argparse


def plot_reward(logfile, min_y, max_y):
    epoch = []
    reward = []
    with open(logfile, 'r') as f:
        for n, line in enumerate(f):
            if n == 0:
                continue  # skip first line
            line = line.strip()
            if line == '':
                continue
            d = json.loads(line)
            epoch.append(int(d['episode']))
            reward.append(float(d['avg_reward_0']))
    if min_y is None:
        min_y = 0
    if max_y is not None:
        plt.ylim([min_y, max_y])
    plt.plot(epoch, reward, label='reward')
    plt.legend()
    plt.savefig('/tmp/out-reward.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()

    parser_ = parsers.add_parser('plot-reward')
    parser_.add_argument('--logfile', type=str, required=True)
    parser_.add_argument('--min-y', type=float)
    parser_.add_argument('--max-y', type=float)
    parser_.set_defaults(func=plot_reward)

    args = parser.parse_args()
    func = args.func
    args_dict = args.__dict__
    del args_dict['func']
    func(**args_dict)
