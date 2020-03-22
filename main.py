from collections import defaultdict

import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from velocity import *
from rnn import *

if __name__ == "__main__":

    keep = [
        "indexFinger.X",
        "indexFinger.Y",
        "indexFinger.Z",
        "hand.X",
        "hand.Y",
        "hand.Z",
        "forearm.X",
        "forearm.Y",
        "forearm.Z",
        "upperArm.X",
        "upperArm.Y",
        "upperArm.Z",
        "rightShoulder.X",
        "rightShoulder.Y",
        "rightShoulder.Z",
    ]

    data = generate_dataset(keep, smooth=True)

    train_data, test_data = train_test_split(
        data, test_size=0.15, shuffle=True, random_state=42
    )

    label2class = {}
    class2label = {}

    for idx, labels in enumerate(np.unique(np.asarray([x[1] for x in data]), axis=0)):
        label2class[str(labels)] = idx
        class2label[idx] = str(labels)

    train_data = [(x, label2class[str(y)]) for (x, y) in train_data]
    test_data = [(x, label2class[str(y)]) for (x, y) in test_data]

    n_epochs = 10
    hidden_size = 200
    # pad_length = max([x.shape[0] for (x,y) in data])
    pad_length = 300
    dimensions = len(keep)
    layers = 2

    model = VelocityNN(hidden_size, dimensions, len(label2class), pad_length, layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    save_file = model_path / "model2"

    train(
        n_epochs,
        model,
        optimizer,
        criterion,
        train_data,
        pad_length,
        dimensions,
        save_file,
    )

    model.load_state_dict(torch.load(save_file))

    # Guesses
    results = [evaluate(model, x, pad_length, dimensions) for x in test_data]

    # Guessed classes
    classes, counts = np.unique(results, return_counts=True)

    # Not guessed classes
    not_guessed = set(class2label.keys()) - set(classes)

    # Not guessed labels
    labels_not_guessed = [class2label[x] for x in not_guessed]

    # Amount of hits
    correct = np.sum([results[idx] == y for (idx, (x, y)) in enumerate(test_data)])

    print(f"Amount of correct guesses was: {correct}")

    counter = defaultdict(int)
    hits = defaultdict(int)
    percentage = {}

    # Count up the amount of points with a particular label
    for (x, y) in test_data:
        counter[y] += 1

    # Count up the hits at a particular label
    for (idx, (x, y)) in enumerate(test_data):
        if results[idx] == y:
            hits[y] += 1

    # Calculate the percentage of hits wrt. amount of test points
    for key in counter.keys():
        try:
            percentage[key] = hits[key] / counter[key]
        except KeyError:
            percentage[key] = 0

    label2hits = np.asarray(
        [
            (((class2label[k])).strip("[]") + f" {v}").split()
            for k, v in percentage.items()
        ]
    ).astype(float)
    xs = list(label2hits[:, 0])
    ys = list(label2hits[:, 1])
    zs = list(label2hits[:, 2])
    intensities = list(label2hits[:, 3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    im = ax.scatter(xs, zs, ys, s=50, edgecolor="black", c=intensities, cmap="PuRd")

    for i in range(len(xs)):
        ax.text(
            xs[i],
            zs[i],
            ys[i],
            f"{list(label2hits[i][:3])}",
            size=8,
            zorder=1,
            color="k",
        )

    fig.colorbar(im)
    ax.set_xlim(left=-1.5, right=1.5)
    ax.set_ylim(bottom=0, top=3.6)
    ax.set_zlim(bottom=0, top=3)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.show()

    import ipdb

    ipdb.set_trace()

