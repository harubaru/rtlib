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

def test_ppm() -> None:
    surface = Surface(128, 128)
    surface.generate_checkerboard(8)
    write_surface("test.ppm", surface)
