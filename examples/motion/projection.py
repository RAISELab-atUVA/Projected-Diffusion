import torch
import math

# TODO: Maybe try working with various shapes?

class Projection:
    
    def __init__(self, cond, acceleration=9.8):
        self.cond = cond
        self.acceleration = acceleration
        self.eps = 0.1
        self.v = 0
        self.height = 0
        
        
    def apply(self, x):
        # Shape = [num_frames, 64, 64]     
        x_p = []

        # threshold = 0.25 #torch.topk(self.cond[0].clone(), k=10, largest=False)[0][-1][-1]
        # position = sorted((self.find_clusters((self.cond[0] > threshold).float())), key=len, reverse=True)
            
        # height = [point[0] for point in position[0]]
        # self.height = 64 - int((min(height) + max(height)) / 2)
        self.height = 56

        # width = [point[1] for point in position[0]]
        width_start = 32 #int((min(width) + max(width)) / 2)

        # Iterate across batches
        for i, x_i in enumerate(x):

            # if i == 2: print(self.height)

            x_i = self.compute_position(x_i.squeeze(0), i, self.height, width_start)
            x_p.append(x_i)
        
        # return x
        return torch.stack(x_p).unsqueeze(1)


    def compute_position(self, x, i, h, w):

        # new_height, self.v = (self.position_change(self.v, 1, h))
        # self.height = new_height
        # new_height = 64 - new_height
        new_height = (64 - h) + int(self.position_change(self.v, i, h) / 2)


        # Real height
        threshold = torch.topk(x.clone().flatten(), k=10, largest=False)[0][-1] + self.eps
        
        position = sorted((self.find_clusters((x > threshold).float())), key=len, reverse=True)
        
        # print(new_height)
        

        in_frame = new_height < 64
        
        
        
        height = [point[0] for point in position[0]]
        avg_height = int((min(height) + max(height)) / 2)
        width = [point[1] for point in position[0]]
        avg_width = int((min(width) + max(width)) / 2)
        
        # print((avg_height - new_height), (w - avg_width))
        
        
        with open("violation-M-120.txt", "a") as file:
            file.write(f"{(avg_height - new_height)} {(w - avg_width)} \n")

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



    # def position_change(self, initial_velocity, acceleration, time, h):
    #     change_in_position = initial_velocity * time + 0.5 * acceleration * time ** 2

    #     return change_in_position

    # Position change with bounce
    def position_change(self, initial_velocity, time, current_position):
        acc = self.acceleration
        
        position_change = 0 + (0.5 * acc * time **2)
#         time_step = 0.1  # Smaller time step for more precision

#         new_velocity = initial_velocity
#         new_position = current_position

#         for t in range(int(time / time_step)):
#             # Update velocity and position
#             new_velocity += acc * time_step
#             new_position -= new_velocity * time_step + 0.5 * acc * time_step ** 2

#             # Check for collision with the bottom of the frame
#             if new_position <= 0:
#                 new_position = -new_position  # Reflect position for bounce
#                 new_velocity = -new_velocity * 0.7  # Apply energy loss

#             # Check for collision with the top of the frame
#             elif new_position >= 64:
#                 new_position = 2 * 64 - new_position  # Reflect position for bounce
#                 new_velocity = -new_velocity * 0.7  # Apply energy loss

#         return int(new_position), new_velocity
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