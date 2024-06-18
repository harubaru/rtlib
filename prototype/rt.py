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

        return ray_origin, ray_direction

class Plane:
    def __init__(self, point, normal):
        self.point = point
        self.normal = normal
    
    def intersect(self, ray_origin, ray_direction) -> bool:
        denom = np.dot(ray_direction, self.normal)
        
        if denom < 1e-8:
            return False
        
        dist = np.dot(self.point - ray_origin, self.normal) / np.dot(ray_direction, self.normal)


class Sphere:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def intersect(self, ray_origin, ray_direction) -> bool:
        oc = self.center - ray_origin
        a = np.dot(ray_direction, ray_direction)
        b = -2.0 * np.dot(ray_direction, oc)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        return discriminant >= 0

# test funcs

def test_ppm() -> None:
    surface = Surface(128, 128)
    surface.generate_checkerboard(8)
    write_surface("test.ppm", surface)

def test_rt():
    surface = Surface(128, 128)
    camera = Camera(surface)
    sphere = Sphere(center=np.array([0.0, 0.0, -1.0]), radius=0.5)

    for y in range(surface.h):
        for x in range(surface.w):
            o, d = camera.ray(x, y)
            if (sphere.intersect(o, d)):
                surface.encode_pixel(x, y, 1.0, 0.0, 0.0)
            else:
                surface.encode_pixel(x, y, 0.0, 0.0, 0.0)

    write_surface("test.ppm", surface)