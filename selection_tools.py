import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import json


class RectXY:
    def __init__(self, fig, ax, width=20, height=20, filepath='test.png'):
        self.fixed_color = 'r'
        self.active_color = 'g'
        self.fig = fig
        self.ax = ax
        self.width = width
        self.height = height
        self.list_of_rects = []
        self._fixed = False
        self.current_rect_index = None
        self.renderer = fig.canvas.get_renderer()
        self.linewidth = 1.0
        self.roi_data_filepath = filepath + '.json'
        self._load_roi_data()
        self.size_change_delta = 10

    def _load_roi_data(self):
        self.list_of_rects = []
        if not os.path.exists(self.roi_data_filepath):
            return
        with open(self.roi_data_filepath, 'r') as f:
            data = json.load(f)
        for d in data:
            x, y = d['x'], d['y']
            height = d['height']
            width = d['width']
            self._add_rect(x, y, width=width, height=height, fixed=True)
        if len(data) > 0:
            self.height = height
            self.width = width

    def populate_roi_data(self, roi_data):
        self.list_of_rects = []
        if len(roi_data) == 0:
            return
        for d in roi_data:
            height = d['height']
            width = d['width']
            x_c, y_c = d['x_center'], d['y_center']
            x, y = self._limit_movement(x_c, y_c, width, height)
            self._add_rect(x, y, width=width, height=height, fixed=True)

            self.height = height
            self.width = width

    def _add_rect(self, x0, y0, width=None, height=None, fixed=True):
        color = self.fixed_color if fixed else self.active_color
        if None in (width, height):
            width, height = self.width, self.height
        rect = matplotlib.patches.Rectangle((x0, y0), width, height,
                                            fill=False, linewidth=self.linewidth, color=color)
        self.list_of_rects.append(rect)
        self.ax.add_patch(rect)
        self.current_rect_index = len(self.list_of_rects) - 1
        self._fixed = fixed

    def _limit_movement(self, x, y, width, height):
        x, y = int(x), int(y)
        y_min, y_max = self.ax.axes.viewLim.p0[0] + 1, self.ax.axes.viewLim.p0[1] - height
        x_min, x_max = self.ax.axes.viewLim.p1[1] + 1, self.ax.axes.viewLim.p1[0] - width
        x_rect = int(np.clip(x - width // 2, a_min=x_min, a_max=x_max))
        y_rect = int(np.clip(y - height // 2, a_min=y_min, a_max=y_max))
        return x_rect, y_rect

    def move(self, event, color=None):
        color = self.active_color if color is None else color
        x, y = event.xdata, event.ydata
        if None in (x, y) or self._fixed or len(self.list_of_rects) == 0 or self.current_rect_index is None:
            return

        current_rect = self.list_of_rects[self.current_rect_index]
        current_rect.set_color(color)  # set active color
        width, height = current_rect.get_width(), current_rect.get_height()
        current_rect.xy = self._limit_movement(x, y, width, height)
        self.fig.canvas.draw_idle()  # update canvas with new rectangle position

    def fix_or_release(self, event):
        if len(self.list_of_rects) == 0:
            return

        if not self._fixed and self.current_rect_index is not None:
            current_rect = self.list_of_rects[self.current_rect_index]

            # Fix box in position
            self._fixed = True
            current_rect.set_color(self.fixed_color)
            self.fig.canvas.draw_idle()
            return

        # Release box only if the pointer within the box bounds
        x, y = int(event.xdata), int(event.ydata)
        for i, rect in enumerate(self.list_of_rects):  # loop through all rectangles
            if self._within_box(x, y, rect):
                self._fixed = False
                self.current_rect_index = i
                break
        self.move(event, color=self.active_color)  # update rectangle position and set the new color

    def _within_box(self, x, y, rect, center_fraction=0.2):
        xb, yb = self._box_center(rect)
        width, height = rect.get_width(), rect.get_height()
        x_within = (xb - width * center_fraction) < x < (xb + width * center_fraction)
        y_within = (yb - height * center_fraction) < y < (yb + height * center_fraction)
        return x_within and y_within

    def _box_center(self, rect):
        x0, y0 = rect.xy
        xb = x0 + rect.get_width() // 2
        yb = y0 + rect.get_height() // 2  # center coordinates of box
        return xb, yb

    def insert_or_delete_rectangle(self, event):
        delete_rectangle = not self._fixed and len(self.list_of_rects) > 0 and self.current_rect_index is not None
        if delete_rectangle:
            self.list_of_rects[self.current_rect_index].set_visible(False)
            self.list_of_rects[self.current_rect_index].remove()  # remove rectangle from axes
            self.list_of_rects.pop(self.current_rect_index)  # remove from internal list
            self.current_rect_index = None
            self.fig.canvas.draw_idle()
            return

        # Insert rectangle
        x, y = event.xdata, event.ydata
        if None in (x, y):
            return  # avoid trying to insert rectangle when pointer is outside active area
        xr, yr = self._limit_movement(x, y, self.width, self.height)
        self._add_rect(xr, yr, fixed=False)
        self.fig.canvas.draw_idle()

    def _convert_to_dict(self):
        list_of_dicts = []
        for rect in self.list_of_rects:
            x, y = rect.get_xy()
            height = rect.get_height()
            width = rect.get_width()
            list_of_dicts.append({'x': x, 'y': y, 'height': height, 'width': width})
        return list_of_dicts

    def save_roi_data_and_close(self):
        roi_dict = self._convert_to_dict()
        with open(self.roi_data_filepath, 'w') as f:
            json.dump(roi_dict, f, indent=4)
        print(json.dumps(roi_dict, indent=4))
        plt.close(self.fig)

    def change_roi_size(self, width_change, height_change):
        current_rect = self.list_of_rects[self.current_rect_index]
        x, y = current_rect.xy
        width, height = current_rect.get_width(), current_rect.get_height()
        width = max(width + width_change, 0)
        height = max(height + height_change, 0)
        x = int(x - width_change / 2)
        y = int(y - height_change / 2)
        current_rect.set_width(width)
        current_rect.set_height(height)
        current_rect.set_xy((x, y))
        self.fig.canvas.draw_idle()

    def json_path(self):
        return self.roi_data_filepath

    def keypress(self, event):
        if event.key == ' ':
            self.insert_or_delete_rectangle(event)
        if event.key == 'e':
            self.save_roi_data_and_close()
        if event.key == '8':
            self.change_roi_size(0, self.size_change_delta)
        if event.key == '2':
            self.change_roi_size(0, -self.size_change_delta)
        if event.key == '4':
            self.change_roi_size(-self.size_change_delta, 0)
        if event.key == '6':
            self.change_roi_size(self.size_change_delta, 0)


def read_and_crop(im, json_path):
    if not os.path.exists(json_path):
        return [im]
    with open(json_path, 'r') as f:
        roi_data = json.load(f)
    roi_list = []
    if len(roi_data) > 0:
        for r in roi_data:
            roi_list.append(im[r['y']:r['y'] + r['height'], r['x']:r['x'] + r['width']])
        return roi_list
    else:
        return [im]


def crop(im, roi_data):
    roi_list = []
    if len(roi_data) > 0:
        for r in roi_data:
            x_c, y_c = r['x_center'], r['y_center']
            width, height = r['width'], r['height']
            y_min, y_max = 0 + 1, im.shape[0] - height
            x_min, x_max = 0 + 1, im.shape[1] - width
            x_r = int(np.clip(x_c - width // 2, a_min=x_min, a_max=x_max))
            y_r = int(np.clip(y_c - height // 2, a_min=y_min, a_max=y_max))
            roi_list.append(im[y_r:y_r + height, x_r:x_r + width])
        return roi_list
    else:
        return [im]


def test():
    import selection_tools

    im_file = 'televinken.jpg'
    image = plt.imread(im_file)
    image = image[:, :, :3]

    # Create a figure for plotting the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title('Insert/delete ROI:s with space bar, \nfix/release ROI:s with left mouse clock, \n'
                 'save ROIs and close window with "e"', fontsize=9)

    print(f'image.shape: {image.shape}')

    r = selection_tools.RectXY(fig, ax, filepath=im_file, width=80, height=80)
    fig.canvas.mpl_connect('motion_notify_event', r.move)
    fig.canvas.mpl_connect('button_press_event', r.fix_or_release)
    fig.canvas.mpl_connect('key_release_event', r.keypress)

    plt.show()

    json_path = im_file + '.json'

    with open(json_path, 'r') as f:
        roi_data = json.load(f)
    im_rois = selection_tools.read_and_crop(image, json_path)
    for im_roi, r in zip(im_rois, roi_data):
        plt.figure()
        title = f'x, y = {r["x"]}:{r["x"] + r["width"]}, {r["y"]}:{r["y"] + r["height"]}'
        plt.imshow(im_roi, cmap='gray')
        plt.title(title)
    plt.show()


if __name__ == '__main__':
    test()
