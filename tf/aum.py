from tensorflow import squeeze, gather
import tensorflow as tf
import tensorflow.compat
from tensorflow.random import normal
import tensorflow.math as tfmath
from dataclasses import dataclass

@dataclass
class AUMRecord:
    sample_id: int
    visits: int
    # target_ndx: int
    target_val: float
    # pred_ndx: int
    pred_val: float
    margin: float
    aum: float

class AUMTracker:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.records = []
        self.sums = {}
        self.visits = {}

    def update(self, logits, targets, sample_ids):
        targets = gather(targets, 1)

        max_logits = tfmath.reduce_max(logits)
        # max_logits = squeeze(max_logits, axis = -1)

        updated = {}

        margins = targets - max_logits

        # for i in range(len(sample_ids)):
        #     sample_id = sample_ids[i].ref()
        #     margin = margins[i]
        #
        #     if sample_id in self.sums:
        #         self.sums[sample_id] += margin
        #         self.visits[sample_id] += 1
        #     else:
        #         self.sums[sample_id] = margin
        #         self.visits[sample_id] = 1
        #
        #     a = self.sums[sample_id] / self.visits[sample_id]
        #
        #     record = AUMRecord(
        #         sample_id,
        #         self.visits[sample_id],
        #         targets[i],
        #         logits[i],
        #         margin,
        #         a,
        #     )
        #     updated[sample_id] = record
        #
        #     if self.verbose:
        #         self.records.append(record)

        for i, (margin, sample_id) in enumerate(zip(margins, sample_ids)):
            sample_id = sample_id.ref()
            if sample_id in self.sums:
                self.sums[sample_id] += margin
                self.visits[sample_id] += 1
            else:
                self.sums[sample_id] = margin
                self.visits[sample_id] = 1

            record = AUMRecord(
                sample_id,
                self.visits[sample_id],
                targets[i],
                logits[i],
                margin,
                self.sums[sample_id] / self.visits[sample_id],
            )
            updated[sample_id] = record

            if self.verbose:
                self.records.append(record)

        return updated

    def graph(self) -> None:
        if not self.verbose:
            print("Not in verbose mode, did not log anything")
            return

        from matplotlib import pyplot as plt

        fig = plt.figure()

        assert len(self.records) != 0
        plt.plot(self.get_all_scores())
        # print(self.get_all_scores())

        plt.savefig("poop.png")
        print("saved graph")

    def get_all_scores(self):
        scores = []
        for record in self.records:
            scores.append(record.aum)
        return scores

    def clear(self):
        self.records = []
        self.sums = {}
        self.visits = {}

if __name__ == "__main__":
    tracker = AUMTracker()
    tracker.update(normal([1, 4]), normal([1, 4]), [1, 2, 3, 4])
    tracker.graph()
