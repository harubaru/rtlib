import numpy as np

# Surface & PPM Code

class Surface:
    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h
        self.buffer = [[0.0, 0.0, 0.0]] * (w * h)
    
    def blit(self, pixels):
        for y in range(self.h):
            for x in range(self.w):
                idx = (y * self.w) + x
                self.buffer[idx] = pixels[idx]
    
    def encode_pixel(self, x: int, y: int, r: float, g: float, b: float):
        self.buffer[(y * self.w) + x] = [r, g, b]

    def decode_pixel(self, x: int, y: int) -> list:
        return self.buffer[(y * self.w) + x]
    
    def aspect_ratio(self) -> float:
        return self.w/self.h

    def generate_checkerboard(self, square_size):
        for y in range(self.h):
            for x in range(self.w):
                if (x // square_size + y // square_size) % 2 == 0:
                    self.encode_pixel(x, y, 0.0, 0.0, 0.0)  # Black
                else:
                    self.encode_pixel(x, y, 1.0, 1.0, 1.0)  # White

def write_surface(filepath: str, surface: Surface) -> None:
    with open(filepath, "w") as f:
        f.write(f"P3\n{surface.w} {surface.h}\n255\n")

        for y in range(surface.h):
            for x in range(surface.w):
                r, g, b = surface.decode_pixel(x, y)

                ir = int(255.999 * r)
                ig = int(255.999 * g)
                ib = int(255.999 * b)

                f.write(f"{ir} {ig} {ib}\n")

# RT Primitives

class Ray:
    def __init__(
            self,
            origin,
            direction
    ):
        self.origin = origin
        self.direction = direction
    
    def at(self, t: float):
        return self.origin + t * self.direction

class Camera:
    def __init__(
            self,
            surface,
            samples_per_pixel = 10,
            max_depth = 10,
            vfov = 90.0,
            look_from = np.array([0.0, 0.0, 0.0]),
            look_at = np.array([0.0, 0.0, -1]),
            up = np.array([0.0, 1.0, 0.0])
    ):
        self.surface = surface
        self.samples_per_pixel = samples_per_pixel
        self.pixel_sample_scale = 1 / self.samples_per_pixel
        self.max_depth = max_depth
        self.vfov = vfov
        self.look_from = look_from
        self.look_at = look_at
        self.up = up

        theta = vfov * np.pi / 180.0
        h = np.tan(theta/2)
        viewport_height = 2 * h
        viewport_width = viewport_height * self.surface.aspect_ratio()

        w = self.look_from - self.look_at
        w_hat = w / (w**2).sum()**0.5
        u = np.cross(self.up, w)
        u_hat = u / (u**2).sum()**0.5
        v = np.cross(w_hat, u_hat)

        viewport_u = viewport_width * u
        viewport_v = viewport_height * -v

        self.pixel_delta_u = viewport_u / self.surface.w
        self.pixel_delta_v = viewport_v / self.surface.h

        viewport_upper_left = self.look_from - (w) - viewport_u/2 - viewport_v/2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
    
    def ray(self, x, y):
        pixel_sample = self.pixel00_loc + (x * self.pixel_delta_u) + (y * self.pixel_delta_v)

        ray_origin = self.look_from
        ray_direction = pixel_sample - ray_origin

        return Ray(ray_origin, ray_direction)

class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0*a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0*a)
            if t1 < t2 and t1 > 0:
                return t1
            if t2 > 0:
                return t2
        return None

    def bounding_box(self):
        return AABB(self.center - self.radius, self.center + self.radius)

class Plane:
    def __init__(self, point, normal):
        self.point = np.array(point)
        self.normal = np.array(normal)
    
    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if np.abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t >= 0:
                return t
        return None

    def bounding_box(self):
        inf = float('inf')
        return AABB(np.array([-inf, -inf, -inf]), np.array([inf, inf, inf]))

# Acceleration Structures

class BVHNode:
    def __init__(self, objects):
        self.left = None
        self.right = None
        self.box = None

        if len(objects) == 1:
            self.left = self.right = objects[0]
            self.box = objects[0].bounding_box()
        elif len(objects) == 2:
            self.left, self.right = sorted(objects, key=lambda x: x.bounding_box().min()[0])
            self.box = self.bounding_box_union(self.left.bounding_box(), self.right.bounding_box())
        else:
            objects.sort(key=lambda x: x.bounding_box().min()[0])
            mid = len(objects) // 2
            self.left = BVHNode(objects[:mid])
            self.right = BVHNode(objects[mid:])
            self.box = self.bounding_box_union(self.left.box, self.right.box)
    
    def bounding_box_union(self, box1, box2):
        small = np.minimum(box1.min(), box2.min())
        large = np.maximum(box1.max(), box2.max())
        return AABB(small, large)

    def intersect(self, ray):
        if not self.box.hit(ray):
            return None
        hit_left = self.left.intersect(ray) if self.left else None
        hit_right = self.right.intersect(ray) if self.right else None
        if hit_left and hit_right:
            return hit_left if hit_left < hit_right else hit_right
        return hit_left or hit_right

class AABB:
    def __init__(self, minimum, maximum):
        self.minimum = np.array(minimum)
        self.maximum = np.array(maximum)
    
    def min(self):
        return self.minimum
    
    def max(self):
        return self.maximum

    def hit(self, ray):
        t_min = (self.min() - ray.origin) / ray.direction
        t_max = (self.max() - ray.origin) / ray.direction
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        t_near = np.maximum(t1[0], np.maximum(t1[1], t1[2]))
        t_far = np.minimum(t2[0], np.minimum(t2[1], t2[2]))
        return t_near <= t_far

class NaiveAccelerationStructure:
    def __init__(self, objects):
        self.objects = objects

    def intersect(self, ray):
        closest_hit = None
        closest_t = float('inf')
        for obj in self.objects:
            t = obj.intersect(ray)
            if t is not None and t < closest_t:
                closest_hit = t
                closest_t = t
        return closest_hit

class RayTracingPipeline:
    def __init__(self, accel_structure, ray_gen, intersect, any_hit, closest_hit, miss):
        self.accel_structure = accel_structure
        self.ray_gen = ray_gen
        self.intersect = intersect
        self.any_hit = any_hit
        self.closest_hit = closest_hit
        self.miss = miss

    def trace_rays(self, surface, world_size, global_rank):
        h = surface.h
        w = surface.w
        rows_per_rank = h // world_size
        start_row = rows_per_rank * global_rank
        end_row = start_row + rows_per_rank if global_rank < world_size - 1 else h

        for y in range(start_row, end_row):
            for x in range(w):
                ray = self.ray_gen(x, y)
                hit = self.accel_structure.intersect(ray)
                if hit and self.any_hit(ray, hit):
                    color = self.closest_hit(ray, hit)
                else:
                    color = self.miss(ray)
                surface.encode_pixel(x, y, *color)

# testing stuff!!
surface = Surface(400, 200)
camera = Camera(surface)

def ray_gen_function(x, y):
    return camera.ray(x, y)

def intersect_function(ray, obj):
    return obj.intersect(ray)

def any_hit_function(ray, hit):
    return True

def closest_hit_function(ray, hit):
    return [1.0, 0.0, 0.0]  # red color for hits

def miss_function(ray):
    return [0.0, 1.0, 0.0]  # black color for misses

pipeline = RayTracingPipeline(
    BVHNode([
        Sphere([0, 0, -5], 1),
        Sphere([2, 0, -5], 1),
        Plane([0, -1, 0], [0, 1, 0])
    ]),
    ray_gen_function,
    intersect_function,
    any_hit_function,
    closest_hit_function,
    miss_function
)

pipeline.trace_rays(surface, 4, 1)
write_surface("output.ppm", surface)
