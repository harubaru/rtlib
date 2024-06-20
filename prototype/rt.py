import tqdm
import numpy as np
from multiprocessing import Process, Array
from PIL import Image

# Constants


EPSILON = 1e-6


# RT Surface


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
                    self.encode_pixel(x, y, *pixels[idx:idx + 3])

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
        return self.w / self.h


# RT Primitives


class Ray:
    def __init__(
            self,
            origin,
            direction,
            parent_ray = None,
    ):
        self.origin = origin
        self.direction = direction

        self.parent_ray = parent_ray
        self.scattered_ray = None
        self.color = np.array([0.0, 0.0, 0.0])
        self.hit_point = np.array([0.0, 0.0, 0.0])
        self.normal = np.array([0.0, 0.0, 0.0])
        self.hit_object = None
        self.t = float('inf')  # distance to closest hit
        self.depth = 0
        self.hit = False
        self.uv = np.array([0.0, 0.0])
        self.light_contrib = np.array([0.0, 0.0, 0.0])

    def at(self, t: float):
        return self.origin + t * self.direction


class Camera:
    def __init__(
            self,
            width, height,
            vfov=120.0,
            look_from=np.array([0.0, 0.0, 0.0]),
            look_at=np.array([0.0, 0.0, -1]),
            up=np.array([0.0, 1.0, 0.0])
    ):
        self.vfov = vfov
        self.look_from = look_from
        self.look_at = look_at
        self.up = up

        theta = vfov * np.pi / 180.0
        viewport_height = 2 * np.tan(theta / 2)
        viewport_width = viewport_height * (width / height)

        w = self.look_from - self.look_at
        w_hat = w / (w ** 2).sum() ** 0.5
        u = np.cross(self.up, w)
        u_hat = u / (u ** 2).sum() ** 0.5

        viewport_u = viewport_width * u
        viewport_v = viewport_height * -np.cross(w_hat, u_hat)

        self.pixel_delta_u = viewport_u / width
        self.pixel_delta_v = viewport_v / height

        viewport_upper_left = self.look_from - w - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)


    def ray(self, x, y):
        pixel_sample = self.pixel00_loc + (x * self.pixel_delta_u) + (
                    y * self.pixel_delta_v)

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
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = half_b ** 2 - a * c
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
        if np.abs(denom) > EPSILON:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t >= EPSILON:
                hit_point = ray.at(t)
                return t, self.normal, hit_point, self
        return None, None, None, None

    def uv(self, hit_point):
        u = (hit_point[0] - self.point[0]) % 1
        v = (hit_point[2] - self.point[2]) % 1
        return u, v


class Triangle:
    def __init__(self, v0, v1, v2, material, uv0=[0, 0], uv1=[1, 0], uv2=[0, 1]):
        self.v0 = np.array(v0, dtype=np.float64)
        self.v1 = np.array(v1, dtype=np.float64)
        self.v2 = np.array(v2, dtype=np.float64)
        self.material = material
        self.normal = np.cross(self.v1 - self.v0, self.v2 - self.v0)
        self.normal /= np.linalg.norm(self.normal)
        self.uv0 = np.array(uv0, dtype=np.float64)
        self.uv1 = np.array(uv1, dtype=np.float64)
        self.uv2 = np.array(uv2, dtype=np.float64)

    def intersect(self, ray):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < EPSILON:
            return None, None, None, None

        f = 1.0 / a
        s = ray.origin - self.v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None, None, None, None

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None, None, None, None

        t = f * np.dot(edge2, q)
        if t > EPSILON:
            hit_point = ray.at(t)
            return t, self.normal, hit_point, self
        else:
            return None, None, None, None

    def uv(self, hit_point):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        local_hit = hit_point - self.v0
        dot00 = np.dot(edge1, edge1)
        dot01 = np.dot(edge1, edge2)
        dot11 = np.dot(edge2, edge2)
        dot20 = np.dot(local_hit, edge1)
        dot21 = np.dot(local_hit, edge2)
        denom = dot00 * dot11 - dot01 * dot01
        v = (dot11 * dot20 - dot01 * dot21) / denom
        w = (dot00 * dot21 - dot01 * dot20) / denom
        u = 1.0 - v - w
        return u * self.uv0 + v * self.uv1 + w * self.uv2


class Mesh:
    def __init__(self, vertices, indices, uvs, material):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.indices = np.array(indices, dtype=np.int32)
        self.uvs = np.array(uvs, dtype=np.float64)
        self.material = material
        self.triangles = []
        for i in range(0, len(indices), 3):
            self.triangles.append(
                Triangle(
                    self.vertices[indices[i]],
                    self.vertices[indices[i + 1]],
                    self.vertices[indices[i + 2]],
                    material,
                    self.uvs[indices[i]],
                    self.uvs[indices[i + 1]],
                    self.uvs[indices[i + 2]]
                )
            )

    def intersect(self, ray):
        hit_info = None
        for triangle in self.triangles:
            t, normal, hit_point, hit_object = triangle.intersect(ray)
            if t is not None and (hit_info is None or t < hit_info[0]):
                hit_info = (t, normal, hit_point, hit_object)
        if hit_info:
            return hit_info
        else:
            return None, None, None, None

    def uv(self, hit_point):
        for triangle in self.triangles:
            t, normal, hp, ho = triangle.intersect(Ray(hit_point - EPSILON * normal, normal))
            if t is not None:
                return triangle.uv(hit_point)
        return 0.0, 0.0


# Acceleration Structures
class BLAS:
    def __init__(self, objects, intersect_fn):
        self.objects = objects
        self.intersect_fn = intersect_fn

    def __call__(self, ray):
        self.intersect_fn(self.objects, ray)


class TLAS:
    def __init__(self, blas_instances, intersect_fn):
        self.blas_instances = blas_instances
        self.intersect_fn = intersect_fn

    def __call__(self, ray):
        self.intersect_fn(self.blas_instances, ray)


class RayTracingPipelineArgs:
    def __init__(self, max_depth=1, samples_per_pixel=4, num_passes=1):
        self.max_depth = max_depth
        self.samples_per_pixel = samples_per_pixel
        self.num_passes = num_passes


class RayTracingPipeline:
    def __init__(self, accel_structure, lights, ray_gen, any_hit, closest_hit, miss, post_process,
                 args):
        self.accel_structure = accel_structure  # our tlas
        self.lights = lights
        self.ray_gen = ray_gen
        self.any_hit = any_hit
        self.closest_hit = closest_hit
        self.miss = miss
        self.post_process = post_process
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
                for _ in range(self.args.num_passes):
                    pass_color = np.array([0.0, 0.0, 0.0])
                    for _ in range(self.args.samples_per_pixel):
                        ray = self.ray_gen(x, y)
                        pass_color += self.trace_ray(ray, self.args.max_depth)
                    pass_color /= self.args.samples_per_pixel
                    color_buffer += pass_color
                average_color = color_buffer / self.args.num_passes
                average_color = self.post_process(x, y, average_color)
                surface.encode_pixel(x, y, *average_color)

    def trace_ray(self, ray, depth):
        if depth == 0:
            return np.array([0, 0, 0])  # we've gone too deep! bail out!

        self.accel_structure(ray)

        if ray.hit and self.any_hit(ray):
            self.closest_hit(ray, self.lights, self.accel_structure)
            if ray.scattered_ray is not None:
                return ray.color * self.trace_ray(ray.scattered_ray, depth - 1)  # recurse!
            else:
                return ray.color  # it's over! return the color!
        else:
            return self.miss(ray)  # no one was hit! return the miss color!


# RT Materials

class Material:
    def scatter(self, ray):
        raise NotImplementedError("Scatter function must be implemented by subclasses")


class Phong(Material):
    def __init__(self, ambient, diffuse, specular, shininess, light_dir):
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.light_dir = light_dir / np.linalg.norm(light_dir)

    def scatter(self, ray):
        view_dir = -ray.direction / np.linalg.norm(ray.direction)
        reflect_dir = reflect(-self.light_dir, ray.normal)

        ambient = self.ambient

        diff = max(np.dot(ray.normal, self.light_dir), 0.0)
        diffuse = self.diffuse * diff

        spec = np.power(max(np.dot(view_dir, reflect_dir), 0.0), self.shininess)
        specular = self.specular * spec

        ray.color = ambient + diffuse + specular
        return True


class Lambertian(Material):
    def __init__(self, albedo, texture=None):
        self.albedo = np.array(albedo)
        self.texture = texture

    def scatter(self, ray):
        ray.normal = ray.normal if np.dot(ray.direction, ray.normal) < 0 else -ray.normal
        scatter_direction = ray.normal + random_in_hemisphere(ray.normal)
        ray.scattered_ray = Ray(ray.hit_point, scatter_direction, ray)

        if self.texture:
            u, v = ray.hit_object.uv(ray.hit_point)
            ray.color = self.texture.decode_texel(u, v)
        else:
            ray.color = self.albedo
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

    def scatter(self, ray):
        reflected = reflect(ray.direction / np.linalg.norm(ray.direction), ray.normal)
        scattered = reflected + self.fuzz * random_unit_vector()
        ray.scattered_ray = Ray(ray.hit_point, scattered, ray)
        if self.texture:
            u, v = ray.hit_object.uv(ray.hit_point)
            ray.color = self.texture.decode_texel(u, v)
        else:
            ray.color = self.albedo # TODO: texture mixing
        return np.dot(ray.scattered_ray.direction, ray.normal) > 0


class Dielectric(Material):
    def __init__(self, refr_idx):
        self.refr_idx = refr_idx

    def scatter(self, ray):
        attenuation = np.array([1.0, 1.0, 1.0])
        refraction_ratio = self.refr_idx if ray.hit else 1.0 / self.refr_idx

        unit_direction = ray.direction / np.linalg.norm(ray.direction)
        cos_theta = min(np.dot(-unit_direction, ray.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)

        cannot_refract = refraction_ratio * sin_theta > 1.0
        if cannot_refract or reflectance(cos_theta, refraction_ratio) > np.random.rand():
            direction = np.array(reflect(unit_direction, ray.normal))
        else:
            direction = np.array(refract(unit_direction, ray.normal, refraction_ratio))

        ray.scattered_ray = Ray(ray.hit_point, direction, ray)
        ray.color = attenuation
        return True


def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 ** 2
    return r0 + (1 - r0) * (1 - cosine) ** 5


def reflect(v, n):
    return v - 2 * np.dot(v, n) * n


def refract(uv, n, etai_over_etat):
    cos_theta = np.dot(-uv, n)
    r_out_parallel = etai_over_etat * (uv + cos_theta * n)
    r_out_perp = -np.sqrt(1.0 - (r_out_parallel ** 2).sum()) * n
    return r_out_parallel + r_out_perp


# RT Lighting

class Light:
    def calculate_intensity(self, hit_point, normal, accel_structure):
        raise NotImplementedError


class PointLight(Light):
    def __init__(self, position, intensity, color):
        self.position = np.array(position)
        self.intensity = intensity
        self.color = np.array(color)

    def calculate_intensity(self, hit_point, normal, accel_structure):
        direction_to_light = self.position - hit_point
        distance_to_light = np.linalg.norm(direction_to_light)
        direction_to_light /= distance_to_light

        shadow_ray = Ray(hit_point + EPSILON * normal, direction_to_light)
        accel_structure(shadow_ray)

        if shadow_ray.hit and shadow_ray.t < distance_to_light:
            return np.zeros(3)  # In shadow
        else:
            diffuse_intensity = max(np.dot(normal, direction_to_light), 0.0)
            return self.intensity * self.color * diffuse_intensity


class DirectionalLight(Light):
    def __init__(self, direction, intensity, color):
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.intensity = intensity
        self.color = np.array(color)

    def calculate_intensity(self, hit_point, normal, accel_structure):
        shadow_ray = Ray(hit_point + EPSILON * normal, -self.direction)
        accel_structure(shadow_ray)

        if shadow_ray.hit:
            return np.zeros(3)  # In shadow
        else:
            diffuse_intensity = max(np.dot(normal, -self.direction), 0.0)
            return self.intensity * self.color * diffuse_intensity


def calculate_lighting(lights, hit_point, normal, accel_structure):
    color = np.zeros(3)
    for light in lights:
        color += light.calculate_intensity(hit_point, normal, accel_structure)
    return color


# testing stuff!!
surface = Surface(int(256), int(224))
camera = Camera(surface.w, surface.h, 120.0)

camera.look_from = np.array([0.0, 0.0, 2.0])
camera.look_at = np.array([0.0, 0.0, -1.0])


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


def any_hit_function(ray):
    # russian roulette
    if ray.depth > 1:
        if np.random.rand() < 0.5:
            return False
    return True


def closest_hit_function(ray, lights, accel_structure):
    material = ray.hit_object.material
    hit = material.scatter(ray)
    if hit:
        if lights:  # move this to pipeline.
            lighting = calculate_lighting(lights, ray.hit_point, ray.normal, accel_structure)
        else:
            lighting = np.array([1.0, 1.0, 1.0])
        ray.color *= lighting
    else:
        ray.color = [0, 0, 0]
    return ray.color


def miss_function(ray):
    unit_direction = ray.direction / np.linalg.norm(ray.direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])


def post_process(x, y, color):
    return np.clip(color, 0.0, 1.0)


def blas_intersect_fn(objects, ray):
    for obj in objects:
        t, normal, hit_point, hit_object = obj.intersect(ray)
        if t is not None and t < ray.t:
            ray.t = t
            ray.normal = normal
            ray.hit_point = hit_point
            ray.hit_object = hit_object
            ray.hit = True


def tlas_intersect_fn(blas_instances, ray):
    for blas_instance in blas_instances:
        blas_instance(ray)


acceleration_structure = TLAS([BLAS([
    Plane([0, -0.5, 0], [0, 1, 0], Lambertian([0.1, 0.4, 0.1],checker_texture())),
    Sphere([-1, 0, -1], 0.5, Lambertian([0.2, 0.6, 0.8])),
    Sphere([0, 0, -1], 0.5, Dielectric(2.4)),
    Sphere([1, 0, -1], 0.5, Metal([0.8, 0.6, 0.2], 0.5)),
    Sphere([-1, 1, -1], 0.5, Metal([1.0, 0.0, 0.0], 0.1)),
    Sphere([0, 1, -1], 0.5, Lambertian([0.0, 1.0, 0.0])),
    Sphere([1, 1, -1], 0.5, Metal([0.0, 0.0, 1.0], 0.75)),
], blas_intersect_fn)], tlas_intersect_fn)

pipeline = RayTracingPipeline(
    acceleration_structure,
    [DirectionalLight([1, -1, -1], 1.0, [1.0, 1.0, 1.0])],
    ray_gen_function,
    any_hit_function,
    closest_hit_function,
    miss_function,
    post_process,
    RayTracingPipelineArgs(
        max_depth=4,
        samples_per_pixel=32,
        num_passes=1
    ),
)

if __name__ == '__main__':
    world_size = 16
    processes = []

    for i in range(world_size):
        process = Process(target=pipeline.dispatch_rays, args=(surface, world_size, i))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    surface.write("output.png")
