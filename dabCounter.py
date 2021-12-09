"""
Counts the Number of dabs. 
"""


class DabCounter(object):
    def __init__(self, class_name, enter_threshold=7, exit_threshold=4):
        self._class_name = class_name

        # If pose counter pases given threshold, we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # We're either dabbin or we ain't
        self._pose_entered = False

        # Number of times we've dabbed
        self._n_repeats = 0

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        # Get Confidence
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        print(pose_classification)

        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            print("Not Dabbin")
            return self._n_repeats

        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False
            print("Dabbin")

        return self.n_repeats
