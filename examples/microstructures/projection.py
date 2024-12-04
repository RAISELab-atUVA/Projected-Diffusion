import torch
import numpy as np
from math import sqrt, isnan
import cv2
from projected_diffusion.utils import plot_images
import random
import sys
import time 


"""
    Clean up this class. Originally designed for porosity projection, but my intuition is that
    this will be less important when damage projections are finalized. It seems that the sum of
    damage areas will map linearly to porosity, making it unnecessary to directly include.
    
    There are several functions that are not currently being used (and may never be used) that
    I have left in for my own reference, including all the porosity projections and `damage_precise`
    which we've essentially determined is too restrictive for inteded purposes.

"""

class Projection:
    
    """
    Projection: Class used to project onto feasible set (only porosity is implemented)
    
    Arguments:
                      TODO: Expand k to [1, 0]
        k           : Float value [0.5, 0] ratio of pixels in top-k
        threshold   : Boundary between "black" and "white" pixels
        lower_bound : True if projection is for black_pixel_count > k
        img_size    : Length of flattened image tensor
    
    """
    def __init__(self, k, threshold, lower_bound=False, img_size=4096, schedule='linear'):
        
        # Recursive DFS work around
        # sys.setrecursionlimit(img_size)
        
        # Margin from threshold for projected pixels
        # Porosity
        self.eps_start = 0.5
        self.eps_end = 0.1
        self.eps = 0.5
        # Damage
        self.eps_u = 0.5
        # self.eps_u = 0.15
        self.eps_l = 0.5
        # self.eps_l = 0.15
        
        # Schedule
        if schedule == 'linear':
            self.schedule = lambda t, noise_steps: self.linear_schedule(t, noise_steps)
            self.beta = 1.1
        elif schedule == 'quadratic':
            self.schedule = lambda t, noise_steps: self.quadractic_schedule(t, noise_steps)
            self.beta = 0.25
        else:
            raise NotImplementedError
        
        # Projection parameters
        self.threshold_gt = threshold
        self.threshold = threshold
        self.lower_bound = lower_bound
        
        # Convert percentage to pixel count
        self.k_perc = k
        self.img_size = img_size
        self.k = img_size 
        self.schedule(1, 1000)
        
        # Damaged pixels
        self.damage = False
        self.damage_tensor = None
        self.damage_tensor_mask = None
        self.damage_constant = False
        self.restoration = False
        
        # List of tuples specifying stuctures: (area upper bound, area lower bound, diameter)
        self.damage_constraints = None
        
        self.shift = False
        
        
    def apply(self, img_tensor, is_batch=False):
        
        # print('project')
        
        # Check if leading batch dimension
        if len(img_tensor.shape) == 4: is_batch = True
        
        iters = img_tensor.shape[0] if is_batch else 1
        
        stacked_tensor = []
        for _, img in enumerate(img_tensor):
            
            if self.damage: p_img_tensor = self.damage_projection(img)
            else: p_img_tensor = torch.cat([self.porosity_projection(s.unsqueeze(0)) for i, s in enumerate(img)], dim=0)
            stacked_tensor.append(p_img_tensor)
            
        out = torch.stack(stacked_tensor) if is_batch else p_img_tensor
        return out
    
    
    def porosity_projection(self, tensor):
        if self.lower_bound: return torch.stack([self.top_k(tensor_i) for i, tensor_i in enumerate(tensor)]).reshape(tensor.shape)
        return torch.stack([self.bottom_k(tensor_i) for i, tensor_i in enumerate(tensor)]).reshape(tensor.shape)
    
    
    def damage_projection(self, tensor):
        # if self.damage_tensor == None: return tensor
        return torch.stack([self.damage_p(tensor_i) for i, tensor_i in enumerate(tensor)]).reshape(tensor.shape)
        
    
    ####################
    # Porosity Helpers #
    ####################
    
    def bottom_k(self, tensor):
        # Flatten the tensor
        flat_tensor = tensor.flatten()

        # Find the indices and values of elements below the threshold
        below_threshold_indices = torch.where(flat_tensor < self.threshold)[0]
        below_threshold_values = flat_tensor[below_threshold_indices]

        # Sort these values and get the indices of the bottom k
        if below_threshold_indices.numel() >= self.k:
            _, bottom_k_indices = torch.topk(below_threshold_values, self.k, largest=False)
        else:
            return tensor

        # Exclude the bottom k values from modification
        modify_indices = torch.ones_like(below_threshold_values, dtype=torch.bool)
        modify_indices[bottom_k_indices] = False
        indices_to_modify = below_threshold_indices[modify_indices]

        # Increase the values to just above the threshold
        # flat_tensor[indices_to_modify] = self.threshold + self.eps
        
        mask = torch.zeros(flat_tensor.size(), dtype=torch.bool).to('cuda')
        mask[indices_to_modify] = True
        flat_tensor = torch.where(mask, torch.full_like(flat_tensor, self.threshold + self.eps), flat_tensor).to('cuda')

        # Reshape the tensor back to its original shape
        return flat_tensor.reshape(tensor.shape)
    

    def top_k(self, tensor):
        # Flatten the tensor
        flat_tensor = tensor.flatten()

        # Find the indices and values of elements above the threshold
        above_threshold_indices = torch.where(flat_tensor > self.threshold)[0]
        above_threshold_values = flat_tensor[above_threshold_indices]

        # Sort these values and get the indices of the top k
        if above_threshold_indices.numel() >= self.k:
            _, top_k_indices = torch.topk(above_threshold_values, self.k, largest=True)
        else:
            return tensor

        # Exclude the top k values from modification
        modify_indices = torch.ones_like(above_threshold_values, dtype=torch.bool)
        modify_indices[top_k_indices] = False
        indices_to_modify = above_threshold_indices[modify_indices]

        # Lower the values to just below the threshold
        # flat_tensor[indices_to_modify] = self.threshold - self.eps
        
        mask = torch.zeros(flat_tensor.size(), dtype=torch.bool).to('cuda')
        mask[indices_to_modify] = True
        flat_tensor = torch.where(mask, torch.full_like(flat_tensor, self.threshold - self.eps), flat_tensor).to('cuda')

        # Reshape the tensor back to its original shape
        return flat_tensor.reshape(tensor.shape)
    
    
    """
        Scheduling Functions: Specific control of how quickly the feasible set shrinks controlling
            - k
            - threshold
            - eps
    
    """
    def linear_schedule(self, t, noise_steps):
        k_final = (1 - self.k_perc) * (self.img_size) if self.lower_bound else self.k_perc * (self.img_size)  
        k_time  = (self.beta)*((0.5 * (self.img_size)) - (self.k_perc * (self.img_size)))*(t/noise_steps)   
        self.k = min(int( k_final + k_time ), self.img_size)
        self.threshold_schedule(t, noise_steps)
        
    def quadractic_schedule(self, t, noise_steps):
        k_final = (1 - self.k_perc) * (self.img_size) if self.lower_bound else self.k_perc * (self.img_size)  
        k_time  = (self.beta)*((0.5 * (self.img_size)) - (self.k_perc * (self.img_size)))*(torch.log(torch.tensor([t])).item())   
        self.k = min(int( k_final + k_time ), self.img_size)
        self.threshold_schedule(t, noise_steps)
        
    def threshold_schedule(self, t, noise_steps):
        self.threshold = self.threshold_gt -  (t/noise_steps)
        self.eps = max(self.eps_start - 2*((noise_steps - t)/noise_steps), self.eps_end)
    
    
    
    ##################
    # Damage Helpers #
    ##################
    
    """
        damage_p: Projection of damage regions based on the following
        
            - Constraints: tuple of three values (area upper bound, area lower bound, minor axis)
            - ProbabilisticDamage: class used to generate a mask for randomized damage areas
    
    """
    def damage_p(self, tensor):
        
        # FIXME: Parameterize constraints
        # constraints = [(400, 390, 15), (68, 60, 8), (68, 60, 8)] [(1500, 1480, 35), (68, 60, 8), (68, 60, 8)]
        constraints = self.damage_constraints
        
        # Constant for final step
        if not self.damage_constant:
            prob_damage = ProbabilisticDamage(self.shift)
            self.damage_tensor_mask, violation = prob_damage.generate_mask(tensor.unsqueeze(0), constraints, restoration=self.restoration)
            self.damage_constant = not violation
            self.shift = violation
            # TODO: add mask dimension for batching with different damages
            if self.damage_constant: print(f"| mask | Mask size is {(1-self.damage_tensor_mask).sum()}")
        else: violation = False
        
        tensor = tensor.squeeze()
        
        if not violation:
            for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[1]):

                        if tensor[i, j] < (self.threshold + self.eps_u) and self.damage_tensor_mask[i, j] == 1:
                                tensor[i, j] = self.threshold + self.eps_u
                        elif tensor[i, j] > (self.threshold - self.eps_l) and self.damage_tensor_mask[i, j] == 0:
                                tensor[i, j] = self.threshold - self.eps_l
                
        return tensor.unsqueeze(0)
                    
        
        

"""
    With no class variables, it may be a better sylistic choice to convert these to 
    static functions. For now, however, this is not an issue -- just something to 
    consider if we ever want to release this code publicly.
    
    Class only used by `damage_p` to adapt damage regions to constraint set. I think
    we need to look specifically at how to improve `constrain_clusters` to be:
    
        (1) Control AR -- This should be pretty easy; this is just the ration between
            the minor axis and major axis. We will just need to control both height
            and width instead of just one.
            
        (2) Capture Theta -- I think this will be challenging to capture accurately
            as despite my original expectation the width/height analysis does not
            necessarily enforce that theta is 0/90. Creating a rotation will be a
            good start, but we need to go beyond that and consider how to adjust
            the way we add/remove pixels.
        
        (3) A tertiary concern should be improve efficiency of this function as it
            is a bottle neck in the sampling process. This is low priority.

"""

class ProbabilisticDamage:
    """
        This class is designed for generating/adapting a random masks to control the size
        of damage regions in samples.
    
    """
    def __init__(self, shift):
        self.shift = shift
        
    
    
    def find_clusters(self, image):
        clusters = []
        visited = set()

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

    def find_edge_pixels(self, structure, image_shape, directions=[(1, 0), (-1, 0), (0, 1), (0, -1)]):
        edge_pixels = []

        for pixel in structure:
            for dr, dc in directions:
                neighbor = (pixel[0] + dr, pixel[1] + dc)
                if neighbor not in structure and self.is_within_image(neighbor, image_shape):
                    edge_pixels.append(pixel)
                    break

        return edge_pixels
    
    def find_external_edge_pixels(self, structure, image_shape, directions=[(1, 0), (-1, 0), (0, 1), (0, -1)]):
        external_edge_pixels = set()

        for pixel in structure:
            for dr, dc in directions:
                neighbor = (pixel[0] + dr, pixel[1] + dc)
                if neighbor not in structure and self.is_within_image(neighbor, image_shape):
                    external_edge_pixels.add(neighbor)

        return list(external_edge_pixels)

    def is_within_image(self, position, image_shape):
        r, c = position
        rows, cols = image_shape
        return 0 <= r < rows and 0 <= c < cols
    
    
    def add_pixel(self, cluster, image_shape, directions=[(1, 0), (-1, 0), (0, 1), (0, -1)]):
        
        edge_pixels = self.find_edge_pixels(cluster, image_shape, directions)
        if len(edge_pixels) == 0: self.find_edge_pixels(cluster, image_shape, [(1, 0), (-1, 0), (0, 1), (0, -1)])

        pixel_selected = False
        while pixel_selected == False:
            random.shuffle(edge_pixels)
            candidate_pixel = edge_pixels[0]
            for dr, dc in directions:
                neighbor = (candidate_pixel[0] + dr, candidate_pixel[1] + dc)
                if neighbor not in cluster and self.is_within_image(neighbor, image_shape):
                    return neighbor
                
                
    def compute_bounding_box(self, points):
        """
        Compute the bounding box for a set of points.

        Parameters:
        points (list of tuple): List of (x, y) coordinates

        Returns:
        tuple: A tuple representing the bounding box (min_x, min_y, max_x, max_y)
        """
        if not points:
            return None

        min_x = min(points, key=lambda x: x[0])[0]
        min_y = min(points, key=lambda x: x[1])[1]
        max_x = max(points, key=lambda x: x[0])[0]
        max_y = max(points, key=lambda x: x[1])[1]

        return (min_x, min_y, max_x, max_y)
    
    def fill_internal_gaps(self, mask, iterations=1):
        """
        Fill internal gaps in the mask, useful for expanded regions.
        """
        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(mask.numpy(), cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return closing

    def adjust_edges_to_target_area(self, mask, target_area, tolerance=0.01):
        """
        Iteratively adjust non-critical edges of the mask to reach the target area.
        """
        current_area = np.sum(mask)
        kernel = np.ones((5, 5), np.uint8)
        max_iter = 100
        while abs(current_area - target_area) / target_area > tolerance and max_iter > 0:
            max_iter -= 1
            if current_area < target_area:
                # Dilate non-critical edges
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                # Erode non-critical edges
                mask = cv2.erode(mask, kernel, iterations=1)

            current_area = np.sum(mask)

        return torch.from_numpy(mask)
    
    
    def find_non_critical_points(self, points, threshold=0.1):
        """
        Identify non-critical points in the structure.
        """
        curvature = self.calculate_curvature(points)
        centroid = self.calculate_centroid(points)
        distances = np.array([np.linalg.norm(np.array(point) - centroid) for point in points])

        # Identify points with low curvature and far from centroid
        non_critical_points = []
        for i, point in enumerate(points):
            if curvature[i] < threshold and distances[i] > np.median(distances):
                non_critical_points.append(point)

        return non_critical_points

    def calculate_curvature(self, points):
        """
        Calculate the curvature at each point of the structure.
        """
        curvature = []
        for i in range(1, len(points) - 1):
            p1, p2, p3 = np.array(points[i - 1]).astype(float), np.array(points[i]).astype(float), np.array(points[i + 1]).astype(float)
            curvature.append(np.linalg.norm(np.cross(p2-p1, p3-p1)) / np.linalg.norm(p3-p1))
        return np.array([0] + curvature + [0])  # Padding the ends with zero curvature

    def calculate_centroid(self, points):
        """
        Calculate the centroid of the structure.
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        _len = len(points)
        centroid_x = sum(x_coords) / _len
        centroid_y = sum(y_coords) / _len
        return np.array([centroid_x, centroid_y])
    
    def contiguous_points(self, points):
        
        image = torch.ones([1, 64, 64]) 
        
        for row, col in points:
            row, col = int(row), int(col)
            if row > 63 or col > 63 or row < 0 or col < 0: pass
            else: image[:, row, col] = 0
        
        clusters = self.find_clusters(image.clone())
        return len(clusters) == 1

    
    
    def constrain_clusters_simple(self, image, clusters, constraints):
        
        # print("| projection | Iter.")
        
        status = (True, "Constraints success.")

        tmp_eps = 5
        new_clust = []
        status_list = []
        for clust, const in zip(clusters, constraints):
            
            # Dimensions
            major_ax, minor_ax = 0.0, 0.0
            dim_eps = 0.5
            
            ax_eval = lambda ax, prec: (ax < (prec - dim_eps) or ax > (prec + dim_eps)) 
            # print(1)
            max_rotation = 15
            while (ax_eval(minor_ax, const[2]) or ax_eval(major_ax, (const[2]*const[3]))) and max_rotation > 0:
                max_rotation -= 1
                clust, major_ax, minor_ax = self.adjust_structure_size(clust, const[2], (const[2]*const[3]))
                
                
            # Fill in gaps so we have to shrink
            clust += self.find_external_edge_pixels(clust, (64,64))
            
            if self.shift:
                x_val = random.randrange(0, 25)
                y_val = random.randrange(0, 25)
                clust = [(x + x_val, y + y_val) for x, y in clust]
            
       
            max_iter = 500
            tolerance = 1e-5
            while abs(len(clust) - const[0]) > 2 and max_iter > 0:
                max_iter -= 1
                if len(clust) < const[0]:
                    clust += self.find_external_edge_pixels(clust, (64,64))
                else:
                    remove_points = self.find_non_critical_points(clust)
                    if len(remove_points) == 0: 
                        remove_points = self.find_edge_pixels(clust, image[0].shape)
                    if len(remove_points) != 0:
                        random.shuffle(remove_points)
                        clust.remove(remove_points[0])
                    
                max_rotation = 15
                while (ax_eval(minor_ax, const[2]) or ax_eval(major_ax, (const[2]*const[3]))) and max_rotation > 0:
                    max_rotation -= 1
                    # print(f"Before: {clust[0:2]}")
                    clust, major_ax, minor_ax = self.adjust_structure_size(clust, const[2], (const[2]*const[3]))     
                    # print(f"After: {clust[0:2]}")
                    
                
                # Remove duplicate points
                clust = list(set([(int(x[0]), int(x[1])) for x in clust]))
                # Remove points off frame
                clust = [(x, y) for x, y in clust if (x < 64 and y < 64) and (x >= 0 and y >= 0)]
                
                
                if max_rotation == 0: status = (False, "Rotation failed.")
                else: status = (True, "Constraints success.")

                
            # print(f"| projection | major, minor: {major_ax}, {minor_ax}")
            # print(3)
            if not self.contiguous_points(clust): status = (False, "Non-contiguous cluster.")
            if abs(len(clust) - const[0]) > 10:  status = (False, "Size adjustment failed.")
            
            # print(f"Accurate to {abs(len(clust) - const[0]) / const[0]}. {const[1]} == {len(clust)}?")
            
            new_clust += clust
            status_list.append(status)

        return new_clust, status_list


    def calculate_current_width(self, rect):
        """
        Calculate the current width (shorter dimension) of the structure.
        """
        _, (width, height), _ = rect
        return min(width, height)
    
    
#     def adjust_structure_size(self, points, desired_width, desired_major_axis_length=None, verbose=False):
#         """
#         Rescale the structure so that its orthogonal width and optionally its major axis length
#         matches the desired dimensions.
#         """
#         # Convert points to a NumPy array
#         points_array = np.array(points, dtype=np.float32)#.reshape((-1, 1, 2))   
#         # print(points_array.shape)
        
#         # rect = cv2.minAreaRect(points_array)
        
#         # Current dimensions and angle
#         # _, (width, height), angle = rect
        
#         img = np.zeros((64, 64), dtype=np.uint8)
#         rr, cc = polygon(points_array[:, 1], points_array[:, 0])
#         rr = np.clip(rr, 0, 63)
#         cc = np.clip(cc, 0, 63)
#         img[rr, cc] = 1

#         # Label the image
#         label_img = measure.label(img)

#         # Get region properties
#         props = measure.regionprops(label_img)
        
#         if len(props) == 0: 
#             print("| error | No props found.")
#             return points, 0, 0
        
#         width, height = props[0].major_axis_length, props[0].minor_axis_length
#         angle = props[0].orientation

#         points_array = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        
#         # FIXME: preventing division by zero
#         if width == 0: width = 1
#         if height == 0: height = 1
        
#         current_width = min(width, height)
#         current_major_axis_length = max(width, height)
        
#         if verbose: print(f"| projection | major_axis: {current_major_axis_length}, minor_axis: {current_width}")

#         # Scale factors
#         scale_factor_width = desired_width / current_width
#         scale_factor_major_axis = 1
#         if desired_major_axis_length is not None:
#             scale_factor_major_axis = desired_major_axis_length / current_major_axis_length

#         # Calculate the major axis vector
#         angle_rad = np.deg2rad(angle)
#         major_axis_vector = (np.cos(angle_rad), np.sin(angle_rad))

#         # Center of the rectangle
#         # cx, cy = rect[0]
#         cx, cy = props[0].centroid

#         # Rescale points
#         scaled_points = []
#         for point in points_array:
#             px, py = point[0]
#             # Translate to origin
#             dx, dy = px - cx, py - cy

#             # Apply scaling
#             dx, dy = dx * scale_factor_major_axis, dy * scale_factor_width

#             # Rotate back and translate to original center
#             scaled_x = cx + dx * major_axis_vector[0] - dy * major_axis_vector[1]
#             scaled_y = cy + dx * major_axis_vector[1] + dy * major_axis_vector[0]

#             if not isnan(scaled_x) and not isnan(scaled_y): scaled_points.append((int(scaled_x), int(scaled_y)))

#         # rect = cv2.minAreaRect(np.array(scaled_points, dtype=np.float32).reshape((-1, 1, 2)))
#         # _, (width, height), angle = rect
        
#         points_array = np.array(scaled_points, dtype=np.float32)
#         img = np.zeros((64, 64), dtype=np.uint8)
#         rr, cc = polygon(points_array[:, 1], points_array[:, 0])
#         rr = np.clip(rr, 0, 63)
#         cc = np.clip(cc, 0, 63)
#         img[rr, cc] = 1

#         # Label the image
#         label_img = measure.label(img)

#         # Get region properties
#         props = measure.regionprops(label_img)
        
#         if len(props) == 0: 
#             print("| error | No props found. Post-alteration.")
#             return points_array, 0, 0
        
#         width, height = props[0].major_axis_length, props[0].minor_axis_length
        
#         current_width = min(width, height)
#         current_major_axis_length = max(width, height)
        
#         return scaled_points, current_major_axis_length, current_width


    def adjust_structure_size(self, points, desired_width, desired_major_axis_length=None, verbose=False):
        """
        Rescale the structure so that its orthogonal width and optionally its major axis length
        matches the desired dimensions.
        """
        # Convert points to a NumPy array
        points_array = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        rect = cv2.minAreaRect(points_array)

        # Current dimensions and angle
        _, (width, height), angle = rect
        
        # Avoid single pixel situations
        if width == 0 or height == 0:
            return points, max(width, height), min(width, height)
        
        # FIXME: preventing division by zero
        if width == 0: width = 1
        if height == 0: height = 1
        
        current_width = min(width, height)
        current_major_axis_length = max(width, height)
        
        if verbose: print(f"| projection | major_axis: {current_major_axis_length}, minor_axis: {current_width}")

        # Scale factors
        scale_factor_width = desired_width / current_width
        scale_factor_major_axis = 1
        if desired_major_axis_length is not None:
            scale_factor_major_axis = desired_major_axis_length / current_major_axis_length

        # Calculate the major axis vector
        angle_rad = np.deg2rad(angle)
        major_axis_vector = (np.cos(angle_rad), np.sin(angle_rad))

        # Center of the rectangle
        cx, cy = rect[0]

        # Rescale points
        scaled_points = []
        for point in points_array:
            px, py = point[0]
            # Translate to origin
            dx, dy = px - cx, py - cy

            # Apply scaling
            dx, dy = dx * scale_factor_major_axis, dy * scale_factor_width

            # Rotate back and translate to original center
            scaled_x = cx + dx * major_axis_vector[0] - dy * major_axis_vector[1]
            scaled_y = cy + dx * major_axis_vector[1] + dy * major_axis_vector[0]

            if not isnan(scaled_x) and not isnan(scaled_y): scaled_points.append((int(scaled_x), int(scaled_y)))
        
        # Remove duplicates and round
        scaled_points = list(set([(int(x[0]), int(x[1])) for x in scaled_points]))
        rect = cv2.minAreaRect(np.array(scaled_points, dtype=np.float32).reshape((-1, 1, 2)))
        _, (width, height), angle = rect
        
        current_width = min(width, height)
        current_major_axis_length = max(width, height)
        
        return scaled_points, current_major_axis_length, current_width

    
    def generate_mask(self, image, constraints, verbose=True, restoration=False):
        
        # Convert to binary image
        
        status = [(False, "Image unconstrained.")]
        max_iter = 5
        while not all(t[0] for t in status) and max_iter > 0:
            
            max_iter -= 1
            clusters = []
            black_point = 0.0
            # TODO: Determine black_point by bottom k
            
            # Locate damage clusters
            # while len(clusters) < len(constraints) and black_point < 0.6:
                # black_point += 0.01
                # print(black_point)
                
            image = (image > (black_point)).float()
            clusters = self.find_clusters(image.clone())
            sort_clusters = sorted(clusters, key=len, reverse=True)

            if len(constraints) < len(sort_clusters): sort_clusters = sort_clusters[:len(constraints)]


            # Adjust clusters to feasible to create mask
            const_clusters, status = self.constrain_clusters_simple(image, sort_clusters, constraints)
            print("| projection | len(clust) = ", len(const_clusters))
            
        
            edge = []
            if True:
                edge += self.find_external_edge_pixels(const_clusters, (64,64))
                edge += self.find_edge_pixels(const_clusters, (64,64))
            

            all_pixels = const_clusters + edge

            mask = torch.ones([1, 64, 64]) 
            for row, col in all_pixels:
                row, col = int(row), int(col)
                if row > 63 or col > 63 or row < 0 or col < 0: pass
                else:
                    if (row, col) in edge: 
                        mask[:, row, col] = 0.5
                    else:
                        mask[:, row, col] = 0
            
            image = (mask > (0.0)).float()
               
            updated_clusters = self.find_clusters(image.clone())
            
            if len(updated_clusters) == len(constraints):
                status.append((True, "All damage regions present."))
                # print(len(updated_clusters[0]))
                # if abs(len(updated_clusters[0]) - constraints[0][0]) > 10:  status.append((False, f"Final size adjustment failed: {len(updated_clusters[0])}."))
            else:
                status.append((False, f"Not all damage regions present, only {len(self.find_clusters(image.clone()))}."))
            
            # TODO: Add condition to check if clusters fused together
            
        if not all(t[0] for t in status): 
            error = "| warning | Constraints violated."
            for stat in status:
                if not stat[0]: error += f" {stat[1]}"
            if verbose: print(error)
            violation = True

        else:
            violation = False
        
        # plot_images(mask.unsqueeze(0))
        return mask.squeeze(), violation
