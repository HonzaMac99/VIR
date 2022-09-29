import vispy
from vispy import visuals, app
from vispy.scene import SceneCanvas
import numpy as np
from matplotlib import pyplot as plt


class Visual_PCL_seq:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, pcl_paths, offset=0):
        self.paths = pcl_paths
        self.offset = offset
        self.total = len(self.paths)

        self.reset()
        self.update_scan()

        print(f"Control \n "
              f"\t N - Forward \n"
              f"\t B - Backward \n"
              f"\t q - Close")

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor=(.5,.5,.5))
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()


        # laserscan part
        self.scan_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.scan_view, 0, 0)
        self.scan_vis = vispy.scene.visuals.Markers()

        self.scan_view.camera = 'turntable'
        self.scan_view.add(self.scan_vis)
        vispy.scene.visuals.XYZAxis(parent=self.scan_view.scene)


    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):
        # first open data
        pcl_path = self.paths[self.offset]
        self.scan = np.load(pcl_path)

        # then change names
        title = f"Scan {self.offset} out of {self.total}"
        self.canvas.title = title

        range_data = np.copy(self.scan[:, 4])

        viridis_range = np.array((range_data * 255), np.uint8)
        viridis_map = self.get_mpl_colormap("jet")
        viridis_colors = viridis_map[viridis_range]

        self.scan_vis.set_data(self.scan[:,:3],
                               face_color=viridis_colors[..., ::-1],
                               edge_color=viridis_colors[..., ::-1],
                               size=1)

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()

        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0

        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1

        elif event.key == 'S':
            print(f'{self.offset}', end=' ')
        elif event.key == 'F':
            print(f'{self.offset}')

        self.update_scan()

        if event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()


    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()

if __name__ == '__main__':
    import sys
    import glob

    ''' Input argument is the path to root folder with .npy files'''
    paths = sorted(glob.glob(f'{sys.argv[1]}/*.npy'))
    vis = Visual_PCL_seq(paths)
    vis.run()
