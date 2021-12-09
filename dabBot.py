# import os
# import tqdm
import cv2
import numpy as np
import dabClassifier
import fullBodyPoseEmbedder
import Smoothing
import dabCounter
import Vizualizer
import dabUtils
from mediapipe.python.solutions import pose as mp_pose


def main():
    class_name = "dab"
    # Init Tracker.
    # pose_tracker = mp_pose.Pose(upper_body_only=False)
    pose_tracker = mp_pose.Pose()
    # Init Embedder
    pose_embedder = fullBodyPoseEmbedder.FullBodyPoseEmbedder()
    # Init Classifier
    pose_classifier = dabClassifier.DabClassifier(
        # pose_samples_folder=pose_samples_folder,
        pose_samples_folder="./data/good data",
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10,
    )
    # Init EMA Smoothing
    pose_classification_filter = Smoothing.EMADictSmoothing(window_size=10, alpha=0.2)

    # Init Dab Counter
    dab_counter = dabCounter.DabCounter(
        # class_name=class_name,
        class_name=class_name,
        enter_threshold=6,
        exit_threshold=4,
    )

    # Init Visualizer
    pose_classification_visualizer = Vizualizer.DabClassificationVisualizer(
        class_name=class_name,
        plot_y_max=10,
    )

    # open camera
    camera = cv2.VideoCapture(0)
    dab_count = 0
    output_frame = None

    while True:
        camera_input = camera.read()
        cv2.imshow("cam", camera_input[1])
        if cv2.waitKey(1) == 27:
            break

        input_frame = cv2.cvtColor(camera_input[1], cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        if pose_landmarks is not None:
            # Get those Landmarks.
            frame_height, frame_width = input_frame.shape[0], input_frame.shape[1]
            pose_landmarks = np.array(
                [
                    [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    for lmk in pose_landmarks.landmark
                ],
                dtype=np.float32,
            )
            assert pose_landmarks.shape == (
                33,
                3,
            ), "Unexpected landmarks shape: {}".format(pose_landmarks.shape)

            # Classify On Current frame.
            pose_classification = pose_classifier(pose_landmarks)

            # Smooth Classification
            pose_classification_filtered = pose_classification_filter(
                pose_classification
            )

            # we dabbin?
            dab_count = dab_counter(pose_classification_filtered)
        else:
            # No pose -> No classification
            pose_classification = None

            # Keep Adding to maintain correct smoothing.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            dab_count = dab_counter.n_repeats
        print(dab_count)

    cv2.destroyAllWindows()


"""
# ideal
while True:
    // get webcam image
    // Classify the pose
    // print dab or not dab
"""

if __name__ == "__main__":
    main()
