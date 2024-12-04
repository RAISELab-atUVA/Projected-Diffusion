import torch
import math

# TODO: Maybe try working with various shapes?

class Projection:
    
    def __init__(self, acceleration=9.8):
        self.acceleration = acceleration
        self.eps = 0.1
        self.v = 0
        self.height = 0
        
        
    def apply(self, x):
        
        x_p = []
        self.height = 56
        width_start = 32

        # Iterate across batches
        for i, x_i in enumerate(x):

            # if i == 2: print(self.height)

            x_i = self.compute_position(x_i.squeeze(0), i, self.height, width_start)
            x_p.append(x_i)
        
        # return x
        return torch.stack(x_p).unsqueeze(1)


    def compute_position(self, x, i, h, w):

        new_height = (64 - h) + int(self.position_change(self.v, i, h) / 2)

        # Real height
        threshold = torch.topk(x.clone().flatten(), k=10, largest=False)[0][-1] + self.eps
        
        position = sorted((self.find_clusters((x > threshold).float())), key=len, reverse=True)
        
        in_frame = new_height < 64
        
        
        height = [point[0] for point in position[0]]
        avg_height = int((min(height) + max(height)) / 2)
        width = [point[1] for point in position[0]]
        avg_width = int((min(width) + max(width)) / 2)
        
        # Remove noise
        x_p = x.clone()

        if in_frame:
            for (i , j) in position[0]:
                x_p[i, j] = 1.
        
        for i in range(64):
            for j in range(64):
                if math.sqrt((i-new_height)**2 + (w-j)**2) <= 5:
                    x_p[i, j] = 0.0


        return x_p



    def position_change(self, initial_velocity, time, current_position):
        acc = self.acceleration
        position_change = 0 + (0.5 * acc * time **2)
        return position_change

    

    def find_clusters(self, image):
        clusters = []
        visited = set()

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        def dfs_iterative(start_row, start_col):
            stack = [(start_row, start_col)]
            cluster = []

            while stack:
                row, col = stack.pop()
                if (row, col) in visited or image[0, row, col] != 0:
                    continue
                visited.add((row, col))
                cluster.append((row, col))

                if row > 0:
                    stack.append((row - 1, col))
                if row < image.shape[1] - 1:
                    stack.append((row + 1, col))
                if col > 0:
                    stack.append((row, col - 1))
                if col < image.shape[2] - 1:
                    stack.append((row, col + 1))

            return cluster

        for row in range(image.shape[1]):
            for col in range(image.shape[2]):
                if (row, col) not in visited and image[0, row, col] == 0:
                    cluster = dfs_iterative(row, col)
                    clusters.append(cluster)

        return clusters