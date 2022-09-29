import torch
import numpy as np
import glob
import matplotlib.patches
import matplotlib.pyplot as plt

class Lidar_Dataset(torch.utils.data.Dataset):
    def __init__(self, pcl_paths):
        ''' Paths to point clouds '''
        self.paths = pcl_paths

        ''' These values are fixed '''
        self.cell_size = (0.2, 0.2)  # Resolution of the bin in Bird Eye View
        self.x_range = (0, 50)   # Distance to the front
        self.y_range = (-20, 20)  # Distance to the sides

        ''' Position of Ego-Car (Should be in the top-middle) '''
        self.ego = (- self.x_range[0] / self.cell_size[0], - self.y_range[0] / self.cell_size[1])

        ''' Number of channels - Design it as you wish '''
        self.channels = 6

        ''' Constructed shape for 2d Map'''
        self.shape = np.array([(self.x_range[1] - self.x_range[0]) / self.cell_size[0],
                               (self.y_range[1] - self.y_range[0]) / self.cell_size[1],
                                self.channels],
                                dtype=np.int)

        ''' Boundaries of values present in the data '''
        self.z_max = 3.5
        self.z_min = -32.03
        self.intensity_max = 1
        self.intensity_min = 0

    def __getitem__(self, index):
        # Dynamically load the point cloud from path list
        pcl = np.load(self.paths[index])

        # Normalize the point clouds / Bird eye view here

        bev = self.lidar_map(pcl)
        batch = {'bev': torch.tensor(bev[..., :-1], dtype=torch.float).permute(2, 0, 1),  # get the shape Ch, H, W
                 'label': torch.tensor(bev[..., -1], dtype=torch.long),
                 'index': index}

        return batch

    def __len__(self):
        return len(self.paths)

    def lidar_map(self, pcl):
        # Filtering the points outside the 2d map
        mask = self.__filter_geometry(pcl)
        pcl = pcl[mask]

        # # reverse x axis
        # mask = self.__filter_geometry_reverse(pcl)
        # pcl = pcl[mask_2]

        max_z = np.full(self.shape[:2], self.z_min)
        coords_num = np.zeros(self.shape[:2])
        box_variance = np.full(self.shape[:2], -1)


        # Vectorized way to get indices for 2d map by discretizing X and Y coordinate
        xy = np.floor(pcl[:, [0, 1]] / self.cell_size + self.ego).astype('i4')

        for i in range(len(xy)):
            coords_num[xy[i, 0], xy[i, 1]] += 1
            if pcl[i, 2] > max_z[xy[i, 0], xy[i, 1]]:
                max_z[xy[i, 0], xy[i, 1]] = pcl[i, 2]

            if box_variance[xy[i, 0], xy[i, 1]] not in [pcl[i, 4], 2]:
                if box_variance[xy[i, 0], xy[i, 1]] == -1:
                    box_variance[xy[i, 0], xy[i, 1]] = pcl[i, 4]
                else:
                    box_variance[xy[i, 0], xy[i, 1]] = 2


        # len xy = 62624


        # Here, create your data representation with helpful values for detection
        # The simplest solution would be to set values of 1 to bins, where the scans occur
        # As you can imagine, this would be non-sufficient for some cases,
        # where "L" shape bins might be vehicles as well as corners of the building
        # Try to introduce some descriptive geometrical and physical values to the input data from the point cloud
        bev = np.zeros(self.shape)

        # x, y, z, intensity, label
        # Add the input features to Bird Eye View here:
        bev[:, :, 0] = (max_z[:] - self.z_min) / (self.z_max-self.z_min)        # normalised max height
        bev[xy[:, 0], xy[:, 1], 1] = pcl[:, 3]                                  # intensity
        bev[xy[:, 0], xy[:, 1], 2] = 1                                          # point occupancy
        bev[:, :, 3] = coords_num[:]                                            # density of points
        bev[:, :, 4] = np.where(box_variance[:, :] == 2, 1, 0)                  # variance


        # np.set_printoptions(threshold=np.inf)
        # print(bev[:, :, 4])
        # np.set_printoptions(threshold=3)


        # Labels
        label_channel = 5
        # No-Vision class
        bev[:, :, label_channel] = 0
        # Background and Vehicle classes
        bev[xy[:, 0], xy[:, 1], label_channel] = pcl[:, 4] + 1

        return bev

    def __filter_geometry(self, points):
        ''' Filter out points, which do not fit into the 2d Map '''

        mask = (points[:, 0] > self.x_range[0] + 0.01) & (
                points[:, 0] < self.x_range[1] - 0.01) & \
               (points[:, 1] > self.y_range[0] + 0.01) & (
                points[:, 1] < self.y_range[1] - 0.01)

        return mask

    def __filter_geometry_reverse(self, points):
        ''' Filter out points, which do not fit into the 2d Map '''

        mask = (points[:, 0] > -self.x_range[1] + 0.01) & (
                points[:, 0] < -self.x_range[0] - 0.01) & \
               (points[:, 1] > self.y_range[0] + 0.01) & (
                points[:, 1] < self.y_range[1] - 0.01)

        return mask


def labels_with_colour(grid, label_channel=None, save=False):
    # colormap used by imshow
    colors = [(0, 0, 0), (0.6, 0.6, 0.6), (1, 0.1, 0.1)]

    # Your bev grids might be different from this.
    # Here you index the colors with labels from grid.
    # Choose the grid for visualization on your own.
    if label_channel:
        img = np.array(colors)[grid[label_channel]]
    else:
        img = np.array(colors)[grid]
    # create a patch (proxy artist) for every color
    patches = [matplotlib.patches.Patch(color=colors[i], label=['No-Vision', 'Background', 'Vehicles'][i]) for i in range(3)]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.imshow(img)
    if save:
        plt.savefig(save)
    else:
        plt.show()

if __name__ == '__main__':
    ''' Example of visualization of the labels. These are the same as in evaluation '''

    # Eval:
    # paths = sorted(glob.glob('../data/*.npy'))

    # Local:
    paths = sorted(glob.glob('data/trn/*.npy'))

    Dataset = Lidar_Dataset(paths)
    # Choose some point cloud
    batch = Dataset[0]
    # Show labels

    labels_with_colour(batch['label'].detach().numpy())
    # Show input channels
    fig, axs = plt.subplots(2, 3)
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            idx = i*len(axs[i]) + j
            if idx < 5:
                axs[i, j].imshow(batch['bev'][idx].detach().numpy())
            else:
                break
    plt.show()



