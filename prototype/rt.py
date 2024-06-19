import tqdm
import numpy as np
from multiprocessing import Process, Array
from PIL import Image

# Constants

EPSILON = 1e-6

# Surface & PPM Code

class Surface:
    def __init__(self, w, h) -> None:
        self.w = w
        self.h = h
        self.buffer = Array('d', [0.0, 0.0, 0.0] * w * h)
    
    def load(self, filepath: str) -> None:
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            img = img.resize((self.w, self.h))
            pixels = np.array(img).flatten()
            pixels = pixels / 256.0
            for y in range(self.h):
                for x in range(self.w):
                    idx = (y * self.w + x) * 3
                    self.encode_pixel(x, y, *pixels[idx:idx+3])

    def write(self, filepath: str) -> None:
        img = Image.new("RGB", (self.w, self.h))
        pixels = [(0, 0, 0)] * self.w * self.h

        for y in range(self.h):
            for x in range(self.w):
                r, g, b = self.decode_pixel(x, y)
                r, g, b = np.sqrt(r), np.sqrt(g), np.sqrt(b)  # Apply gamma correction
                ir = int(256 * np.clip(r, 0.0, 0.999))
                ig = int(256 * np.clip(g, 0.0, 0.999))
                ib = int(256 * np.clip(b, 0.0, 0.999))

                pixels[y * self.w + x] = (ir, ig, ib)
        
        img.putdata(pixels)
        img.save(filepath)
    
    def blit(self, pixels):
        for y in range(self.h):
            for x in range(self.w):
                idx = (y * self.w) + x
                self.buffer[idx] = pixels[idx]
    
    def encode_pixel(self, x: int, y: int, r: float, g: float, b: float):
        index = (y * self.w + x) * 3
        self.buffer[index] = r
        self.buffer[index + 1] = g
        self.buffer[index + 2] = b

    def decode_pixel(self, x: int, y: int) -> list:
        index = (y * self.w + x) * 3
        return [self.buffer[index], self.buffer[index + 1], self.buffer[index + 2]]
    
    def decode_texel(self, u: float, v: float) -> list:
        x = int(u * self.w)
        y = int(v * self.h)
        return self.decode_pixel(x, y)
    
    def aspect_ratio(self) -> float:
        return self.w/self.h

# RT Primitives

class Payload:
    def __init__(self):
        self.scattered_ray = None
        self.color = np.array([0.0, 0.0, 0.0])
        self.hit_point = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 0.0])
        self.hit_object = None
        self.t = float('inf') # distance to closest hit
        self.depth = 0
        self.hit = False
        self.uv = np.array([0.0, 0.0])

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
            vfov=120.0,
            look_from=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, -1]),
            up=np.array([0.0, 1.0, 0.0])
    ):
        self.surface = surface
        self.vfov = vfov
        self.look_from = look_from
        self.look_at = look_at
        self.up = up

        theta = vfov * np.pi / 180.0
        h = np.tan(theta / 2)
        viewport_height = 2 * h
        viewport_width = viewport_height * self.surface.aspect_ratio()

        w = self.look_from - self.look_at
        w_hat = w / (w ** 2).sum() ** 0.5
        u = np.cross(self.up, w)
        u_hat = u / (u ** 2).sum() ** 0.5
        v = np.cross(w_hat, u_hat)

        viewport_u = viewport_width * u
        viewport_v = viewport_height * -v

        self.pixel_delta_u = viewport_u / self.surface.w
        self.pixel_delta_v = viewport_v / self.surface.h

        viewport_upper_left = self.look_from - (w) - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

    def ray(self, x, y):
        offset = np.random.uniform(-0.5, 0.5, 2)
        pixel_sample = self.pixel00_loc + ((x+offset[0]) * self.pixel_delta_u) + ((y+offset[1]) * self.pixel_delta_v)

        ray_origin = self.look_from
        ray_direction = pixel_sample - ray_origin

        return Ray(ray_origin, ray_direction)

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        half_b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = half_b**2 - a * c
        if discriminant > 0:
            sqrtd = np.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if root < 0:
                root = (-half_b + sqrtd) / a
            if root > EPSILON:
                hit_point = ray.at(root)
                hit_normal = (hit_point - self.center) / self.radius
                return root, hit_normal, hit_point, self
        return None, None, None, None

    def uv(self, hit_point):
        p = (hit_point - self.center) / self.radius
        u = 0.5 + np.arctan2(p[2], p[0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(p[1]) / np.pi
        return u, v

class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point)
        self.normal = np.array(normal)
        self.material = material
    
    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if np.abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t >= EPSILON:
                hit_point = ray.at(t)
                return t, self.normal, hit_point, self
        return None, None, None, None
    
    def uv(self, hit_point):
        u = (hit_point[0] - self.point[0]) % 1
        v = (hit_point[2] - self.point[2]) % 1
        return u, v

# Acceleration Structures
class NaiveAccelerationStructure:
    def __init__(self, objects):
        self.objects = objects

    def intersect(self, ray, payload):
        for obj in self.objects:
            t, hit_normal, hit_point, hit_obj = obj.intersect(ray)
            if t is not None and t < payload.t:
                payload.t = t
                payload.normal = hit_normal
                payload.hit_object = hit_obj
                payload.hit = True
                payload.hit_point = hit_point
                payload.uv = hit_obj.uv(hit_point)

class RayTracingPipelineArgs:
    def __init__(self, max_depth=1, samples_per_pixel=4, jitter_range=0.5, jitter_factor=4.0):
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel
        self.jitter_range = jitter_range
        self.jitter_factor = jitter_factor

class RayTracingPipeline:
    def __init__(self, accel_structure, ray_gen, any_hit, closest_hit, miss, args):
        self.accel_structure = accel_structure
        self.ray_gen = ray_gen
        self.any_hit = any_hit
        self.closest_hit = closest_hit
        self.miss = miss
        self.args = args

    def dispatch_rays(self, surface, world_size, global_rank):
        h = surface.h
        w = surface.w
        rows_per_rank = h // world_size
        start_row = rows_per_rank * global_rank
        end_row = start_row + rows_per_rank if global_rank < world_size - 1 else h

        for y in tqdm.trange(start_row, end_row):
            for x in range(w):
                color_buffer = np.array([0.0, 0.0, 0.0])
                for _ in range(self.args.samples_per_pixel):
                    ray = self.ray_gen(x, y)
                    color_buffer += self.trace_ray(ray, self.args.max_depth)
                average_color = color_buffer / self.args.samples_per_pixel
                surface.encode_pixel(x, y, *average_color)

    def trace_ray(self, ray, depth):
        if depth == 0:
            return np.array([0, 0, 0]) # we've gone too deep! bail out!

        payload = Payload()
        self.accel_structure.intersect(ray, payload)

        if payload.hit and self.any_hit(ray, payload):
            self.closest_hit(ray, payload)
            if payload.scattered_ray is not None:
                return payload.color * self.trace_ray(payload.scattered_ray, depth - 1) # recurse!
            else:
                return payload.color # it's over! return the color!
        else:
            return self.miss(ray, payload) # noone was hit! return the miss color!

# RT Materials

class Material:
    def scatter(self, ray, payload):
        raise NotImplementedError("Scatter function must be implemented by subclasses")

class Lambertian(Material):
    def __init__(self, albedo, texture=None):
        self.albedo = np.array(albedo)
        self.texture = texture

    def scatter(self, ray, payload):
        payload.normal = payload.normal if np.dot(ray.direction, payload.normal) < 0 else -payload.normal
        scatter_direction = payload.normal + random_in_hemisphere(payload.normal)
        payload.scattered_ray = Ray(payload.hit_point, scatter_direction)

        if self.texture:
            u, v = payload.hit_object.uv(payload.hit_point)
            payload.color = self.texture.decode_texel(u, v)
        else:
            payload.color = self.albedo
        return True

def random_in_hemisphere(normal):
    in_unit_sphere = np.random.normal(size=3)
    in_unit_sphere /= np.linalg.norm(in_unit_sphere)
    if np.dot(in_unit_sphere, normal) > 0.0:
        return in_unit_sphere
    else:
        return -in_unit_sphere

def random_unit_vector():
    return random_in_hemisphere(np.array([0, 1, 0]))

class Metal(Material):
    def __init__(self, albedo, fuzz, texture=None):
        self.albedo = np.array(albedo)
        self.fuzz = fuzz if fuzz < 1 else 1
        self.texture = texture
    
    def scatter(self, ray, payload):
        reflected = reflect(ray.direction / np.linalg.norm(ray.direction), payload.normal)
        scattered = reflected + self.fuzz * random_unit_vector()
        payload.scattered_ray = Ray(payload.hit_point, scattered)
        if self.texture:
            u, v = payload.hit_object.uv(payload.hit_point)
            payload.color = self.texture.decode_texel(u, v)
        else:
            payload.color = self.albedo
        return np.dot(payload.scattered_ray.direction, payload.normal) > 0

class Dielectric(Material):
    def __init__(self, refr_idx):
        self.refr_idx = refr_idx
    
    def scatter(self, ray, payload):
        attenuation = np.array([1.0, 1.0, 1.0])
        refraction_ratio = self.refr_idx if payload.hit else 1.0 / self.refr_idx

        unit_direction = ray.direction / np.linalg.norm(ray.direction)
        cos_theta = min(np.dot(-unit_direction, payload.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta**2)

        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta, refraction_ratio) > np.random.rand():
            direction = np.array(reflect(unit_direction, payload.normal))
        else:
            direction = np.array(refract(unit_direction, payload.normal, refraction_ratio))
        
        payload.scattered_ray = Ray(payload.hit_point, direction)
        payload.color = attenuation
        return True
    
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0**2
    return r0 + (1 - r0) * (1 - cosine)**5

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def refract(uv, n, etai_over_etat):
    cos_theta = np.dot(-uv, n)
    r_out_parallel = etai_over_etat * (uv + cos_theta * n)
    r_out_perp = -np.sqrt(1.0 - (r_out_parallel**2).sum()) * n
    return r_out_parallel + r_out_perp


# testing stuff!!
surface = Surface(int(196*8), int(128*8))
camera = Camera(surface)

camera.look_from = np.array([0.0, 0.0, 2.0])
camera.look_at = np.array([0.0, 0.0, -1.0])

indeed = Surface(388, 421)
indeed.load("indeed.jpg")

# checker board texture
def checker_texture() -> Surface:
    tex = Surface(256, 256)
    for y in range(tex.h):
        for x in range(tex.w):
            c = (x // 32) % 2 != (y // 32) % 2
            tex.encode_pixel(x, y, *([1.0, 1.0, 1.0] if c else [0.0, 0.0, 0.0]))
    return tex

def ray_gen_function(x, y):
    return camera.ray(x, y)

def any_hit_function(ray, payload):
    # russian roulette
    if payload.depth > 1:
        if np.random.rand() < 0.5:
            return False
    return True

def closest_hit_function(ray, payload):
    material = payload.hit_object.material
    hit = material.scatter(ray, payload)
    if hit:
        color = payload.color
    else:
        color = [0, 0, 0]
    return color

def miss_function(ray, payload):
    unit_direction = ray.direction / np.linalg.norm(ray.direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])

pipeline = RayTracingPipeline(
    NaiveAccelerationStructure([
        Plane([0, -0.5, 0], [0, 1, 0], Lambertian([0.1, 0.4, 0.1], indeed)),
        Sphere([-1, 0, -1], 0.5, Lambertian([0.2, 0.6, 0.8])),
        Sphere([0, 0, -1], 0.5, Dielectric(2.4)),
        Sphere([1, 0, -1], 0.5, Metal([0.8, 0.6, 0.2], 0.5)),
        # sphere on top of the first sphere
        Sphere([-1, 1, -1], 0.5, Metal([1.0, 0.0, 0.0], 0.1, checker_texture())),
        # sphere on top of the second sphere
        Sphere([0, 1, -1], 0.5, Lambertian([0.0, 1.0, 0.0], checker_texture())),
        # sphere on top of the third sphere
        Sphere([1, 1, -1], 0.5, Metal([0.0, 0.0, 1.0], 0.75, checker_texture()))
    ]),
    ray_gen_function,
    any_hit_function,
    closest_hit_function,
    miss_function,
    RayTracingPipelineArgs(4, 8, 0.5, 1.5)
)

from multiprocessing import Process

if __name__ == '__main__':
    world_size = 32
    processes = []

    for i in range(world_size):
        process = Process(target=pipeline.dispatch_rays, args=(surface, world_size, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    surface.write("output.png")
