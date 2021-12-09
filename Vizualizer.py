""" 
    Keeps track of classifications.
"""
import io
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import requests
from matplotlib import pyplot as plt


class DabClassificationVisualizer(object):
    def __init__(
        self,
        class_name,
        plot_location_x=0.05,
        plot_location_y=0.05,
        plot_max_width=0.4,
        plot_max_height=0.4,
        plot_figsize=(9, 4),
        plot_x_max=None,
        plot_y_max=None,
        counter_location_x=0.85,
        counter_location_y=0.05,
        counter_font_path="https://github.com/googlefonts/robot/blob/main/src/hinted/Robot-Regular.tff?raw=true",
        counter_font_color="red",
        counter_font_size=0.15,
    ):

        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None
        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    """
        Renders pose classification and counter.
    """

    def __call__(
        self,
        frame,
        pose_classification,
        pose_classification_filtered,
        dab_count,
    ):

        # History.
        # probably won't need this.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # output frame
        # output_img = Image.fromarray(frame)
        # output_width = output_img.size[0]
        # output_height = output_img.size[1]
        output_width = 500
        output_height = 500

        # Draw the plot.
        # may not need this.
        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail(
            (
                int(output_width * self._plot_max_width),
                int(output_height * self._plot_max_height),
            ),
            Image.ANTIALIAS,
        )
        ouput_img.paste(
            img,
            (
                int(output_width * self._plot_location_x),
                int(output_height * self._plot_location_y),
            ),
        )

        # draw the count.
        ouput_img_draw = ImageDraw.Draw(output_img)
        if self._counter_font is None:
            font_size = int(output_height * self._counter_font_size)
            font_request = requests.get(self._counter_font_path, allow_redirects=True)
            self._counter_font = ImageFont.truetype(
                io.BytesIO(font_request.content), size=font_size
            )
        output_img_draw.text(
            (
                output_width * self._counter_location_x,
                output_height * self._counter_location_y,
            ),
            str(repetitions_count),
            font=self._counter_font,
            fill=self._counter_font_color,
        )

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [
            self._pose_classification_history,
            self._pose_classification_filtered_history,
        ]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
                    plt.plot(y, linewidth=7)

        plt.grid(axis="y", alpha=0.75)
        plt.xlabel("Frame")
        plt.ylabel("Confidence")
        plt.title("Classification history for `{}`".format(self._class_name))
        plt.legend(loc="upper right")

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]),
        )
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img
