import numpy as np
import os
import csv

"""
Pose Classifier For the Dab pose.
Humbly Liberated from:
https://colab.research.google.com/drive/19txHpN8exWhstO6WVkfmYYVC6uug_oVR#scrollTo=y230jVvP1u33
"""


class DabClassifier(object):
    def __init__(
        self,
        pose_samples_folder,
        pose_embedder,
        file_extension="csv",
        file_separator=",",
        n_landmarks=33,
        n_dimensions=3,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
        axes_weights=(1.0, 1.0, 0.2),
    ):

        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights
        self._pose_samples = self._load_pose_samples(
            pose_samples_folder,
            file_extension,
            file_separator,
            n_landmarks,
            n_dimensions,
            pose_embedder,
        )

    """
    Loads pose samples from a given folder. 
    Not sure if i'll need it, but it's used futher down the line, so we'll see.
    """

    def _load_pose_samples(
        self,
        pose_samples_folder,
        file_extension,
        file_separator,
        n_landmarks,
        n_dimensions,
        pose_embedder,
    ):

        file_names = [
            name
            for name in os.listdir(pose_samples_folder)
            if name.endswith(file_extension)
        ]

        pose_samples = []
        for file_name in file_names:
            class_name = file_name[: -(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    # print (n_landmarks * n_dimensions + 1)
                    # print (row)
                    # assert len(row) == n_landmarks * n_dimensions + 1, 'wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[2:], np.float32).reshape(
                        [n_landmarks, n_dimensions]
                    )
                    pose_samples.append(
                        PoseSample(
                            name=row[0],
                            landmarks=landmarks,
                            # class_name=class_name,
                            class_name=row[1],
                            embedding=pose_embedder(landmarks),
                        )
                    )

            return pose_samples

    """
    Classifies Current Pose.
    """

    def __call__(self, pose_landmarks):
        # Check that given, and target have the same shape.
        assert pose_landmarks.shape == (
            self._n_landmarks,
            self._n_dimensions,
        ), "Unexpected shape: {}".format(pose_landmarks.shape)

        # get pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(
            pose_landmarks * np.array([-1, 1, 1])
        )

        # Filter by max distance.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            max_dist_heap.append([max_dist, sample_idx])
        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[: self._top_n_by_max_distance]

        # Filter by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(
                    np.abs(sample.embedding - flipped_pose_embedding)
                    * self._axes_weights
                ),
            )
            mean_dist_heap.append([mean_dist, sample_idx])
        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[: self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [
            self._pose_samples[sample_idx].class_name
            for _, sample_idx in mean_dist_heap
        ]
        result = {
            class_name: class_names.count(class_name) for class_name in set(class_names)
        }

        return result


"""
Pose Sample
"""


class PoseSample(object):
    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name
        self.embedding = embedding


class PoseSampleOutlier(object):
    def __init_(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes
