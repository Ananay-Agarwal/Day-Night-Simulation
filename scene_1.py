import taichi as ti
import numpy as np
from raytracer import render_kernel

ti.init(arch=ti.cuda, default_fp=ti.f32)

# Window dimensions
WIDTH, HEIGHT = 800, 600
ASPECT_RATIO = WIDTH / HEIGHT

# Scene parameters
FOV = 1.2
MAX_BOUNCES = 7

# Camera control
camera_angle_x = 0.0
camera_angle_y = 0.5
camera_distance = 5.0
last_mouse_pos = (0, 0)

# ------------------------------------------------------------------
# Scene data (all defined as Taichi fields)
# ------------------------------------------------------------------
num_spheres = 4
sphere_centers = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)
sphere_radii = ti.field(dtype=ti.f32, shape=num_spheres)
sphere_albedos = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)
sphere_material_types = ti.field(dtype=ti.i32, shape=num_spheres)   # new
sphere_ior = ti.field(dtype=ti.f32, shape=num_spheres)              # new

ground_y = -1.5

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
light_pos[None] = ti.math.vec3(3.0, 5.0, 2.0)
light_color = ti.math.vec3(1.0, 1.0, 0.9) * 1.5
ambient = ti.math.vec3(0.1, 0.1, 0.15)

# Output buffer
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

def init_scene():
    # Sphere 0: red diffuse
    sphere_centers[0] = ti.math.vec3(-1.2, -0.5, 0.0)
    sphere_radii[0] = 0.8
    sphere_albedos[0] = ti.math.vec3(0.9, 0.2, 0.2)
    sphere_material_types[0] = 0   # diffuse
    sphere_ior[0] = 1.0

    # Sphere 1: green diffuse
    sphere_centers[1] = ti.math.vec3(1.5, -0.2, -1.0)
    sphere_radii[1] = 0.7
    sphere_albedos[1] = ti.math.vec3(0.2, 0.8, 0.2)
    sphere_material_types[1] = 0   # diffuse
    sphere_ior[1] = 1.0

    # Sphere 2: blue diffuse
    sphere_centers[2] = ti.math.vec3(0.0, 0.2, 1.2)
    sphere_radii[2] = 0.6
    sphere_albedos[2] = ti.math.vec3(0.2, 0.3, 1.0)
    sphere_material_types[2] = 0   # diffuse
    sphere_ior[2] = 1.0

    # Sphere 3: yellow diffuse
    sphere_centers[3] = ti.math.vec3(-0.5, -1.0, 2.0)
    sphere_radii[3] = 0.5
    sphere_albedos[3] = ti.math.vec3(0.9, 0.8, 0.2)
    sphere_material_types[3] = 0   # diffuse
    sphere_ior[3] = 1.0

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def main():
    global camera_angle_x, camera_angle_y, camera_distance
    init_scene()

    window = ti.ui.Window("Real-Time Ray Tracer (RTX GPU)", (WIDTH, HEIGHT),
                          vsync=True, show_window=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    last_mouse_x = -1
    last_mouse_y = -1
    mouse_sensitivity = 0.8
    t = 0.0

    def update_camera():
        center = ti.math.vec3(0.0, 0.0, 0.0)
        x = camera_distance * ti.cos(camera_angle_x) * ti.cos(camera_angle_y)
        z = camera_distance * ti.sin(camera_angle_x) * ti.cos(camera_angle_y)
        y = camera_distance * ti.sin(camera_angle_y) + 1.0
        return ti.math.vec3(x, y, z), center

    print("Rendering on GPU...")

    while window.running:
        t += 0.005

        # Animate light
        radius = 4.0
        light_x = ti.math.sin(t) * radius
        light_z = ti.math.cos(t * 0.7) * radius
        light_y = 3.5 + ti.math.sin(t * 1.3) * 1.2
        light_pos[None] = ti.math.vec3(light_x, light_y, light_z)

        camera_distance = max(2.0, min(12.0, camera_distance))

        if window.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = window.get_cursor_pos()
            if last_mouse_x != -1 and last_mouse_y != -1:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                camera_angle_x += dx * mouse_sensitivity
                camera_angle_y += dy * mouse_sensitivity
                camera_angle_y = max(-1.48, min(1.48, camera_angle_y))
            last_mouse_x, last_mouse_y = mouse_x, mouse_y
        else:
            last_mouse_x, last_mouse_y = -1, -1

        cam_pos, cam_target = update_camera()

        # Call the generic render kernel with the new material arrays
        render_kernel(pixels,
                      cam_pos, cam_target,
                      FOV, ASPECT_RATIO, MAX_BOUNCES,
                      sphere_centers, sphere_radii, sphere_albedos,
                      sphere_material_types, sphere_ior, num_spheres,
                      ground_y,
                      light_pos[None], light_color, ambient)

        canvas.set_image(pixels)
        window.show()

        if window.is_pressed(ti.ui.ESCAPE):
            break

if __name__ == "__main__":
    main()