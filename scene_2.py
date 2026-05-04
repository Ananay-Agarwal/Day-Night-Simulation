import taichi as ti
import numpy as np
from raytracer import render_kernel

ti.init(arch=ti.cuda, default_fp=ti.f32)

# Window dimensions
WIDTH, HEIGHT = 800, 600
ASPECT_RATIO = WIDTH / HEIGHT

# Scene parameters
FOV = 1.2
MAX_BOUNCES = 15

# Camera control
camera_angle_x = 0.0
camera_angle_y = 0.4          # look down slightly
camera_distance = 5.0
last_mouse_pos = (0, 0)

# ------------------------------------------------------------------
# Scene data (all defined as Taichi fields)
# ------------------------------------------------------------------
num_spheres = 6
sphere_centers = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)
sphere_radii = ti.field(dtype=ti.f32, shape=num_spheres)
sphere_albedos = ti.Vector.field(3, dtype=ti.f32, shape=num_spheres)
sphere_material_types = ti.field(dtype=ti.i32, shape=num_spheres)
sphere_ior = ti.field(dtype=ti.f32, shape=num_spheres)

ground_y = -3.0

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
light_pos[None] = ti.math.vec3(3.0, 5.0, 2.0)
light_color = ti.math.vec3(1.0, 1.0, 0.9) * 1.5
ambient = ti.math.vec3(0.2, 0.2, 0.2)

# Output buffer
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

def init_scene():
    # Sphere 0: diffuse red
    sphere_centers[0] = ti.math.vec3(-1.2, -0.5, 0.0)
    sphere_radii[0] = 0.8
    sphere_albedos[0] = ti.math.vec3(0.9, 0.2, 0.2)
    sphere_material_types[0] = 0   # diffuse
    sphere_ior[0] = 1.0

    # Sphere 1: mirror (green tinted)
    sphere_centers[1] = ti.math.vec3(1.5, -0.2, -1.0)
    sphere_radii[1] = 0.7
    sphere_albedos[1] = ti.math.vec3(0.2, 0.8, 0.2)
    sphere_material_types[1] = 1   # mirror
    sphere_ior[1] = 1.0

    # Sphere 2: glass (blue, IOR 1.5)
    sphere_centers[2] = ti.math.vec3(0.0, 0.2, 1.2)
    sphere_radii[2] = 0.6
    sphere_albedos[2] = ti.math.vec3(0.2, 0.3, 1.0)
    sphere_material_types[2] = 2   # dielectric
    sphere_ior[2] = 1.5

    # Sphere 3: diffuse yellow
    sphere_centers[3] = ti.math.vec3(-0.5, -0.8, 2.0)
    sphere_radii[3] = 0.5
    sphere_albedos[3] = ti.math.vec3(0.9, 0.8, 0.2)
    sphere_material_types[3] = 0   # diffuse
    sphere_ior[3] = 1.0

    # Sphere 4: Sun (bright orange-yellow)
    sphere_centers[4] = ti.math.vec3(1000.0, 1000.0, 1000.0)
    sphere_radii[4] = 0.0
    sphere_albedos[4] = ti.math.vec3(1.0, 0.7, 0.3)  # warm orange/yellow
    sphere_material_types[4] = 3  # EMISSIVE
    sphere_ior[4] = 1.0

    # Sphere 5: Moon (pale gray with slight blue)
    sphere_centers[5] = ti.math.vec3(1000.0, 1000.0, 1000.0)
    sphere_radii[5] = 0.0
    sphere_albedos[5] = ti.math.vec3(0.9, 0.95, 1.0)  # cool off-white
    sphere_material_types[5] = 3
    sphere_ior[5] = 1.0

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

    # Pause state
    paused = False
    space_prev = False

    def update_camera():
        center = ti.math.vec3(0.0, 0.0, 0.0)
        x = camera_distance * ti.cos(camera_angle_x) * ti.cos(camera_angle_y)
        z = camera_distance * ti.sin(camera_angle_x) * ti.cos(camera_angle_y)
        y = camera_distance * ti.sin(camera_angle_y) + 1.0
        return ti.math.vec3(x, y, z), center

    def smoothstep(edge0, edge1, x):
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    print("Rendering on GPU with day/night simulation...")

    while window.running:
        # ===== Toggle pause with spacebar =====
        space_pressed = window.is_pressed(ti.ui.SPACE)
        if space_pressed and not space_prev:
            paused = not paused
        space_prev = space_pressed

        if not paused:
            t += 0.005

            # ===== Day / Night cycle =====
            sun_angle = t * 0.6
            radius = 6.0
            y_amp = 5.0
            light_x = ti.math.cos(sun_angle) * radius
            light_z = ti.math.sin(sun_angle) * radius * 0.8
            light_y = ti.math.sin(sun_angle) * y_amp
            light_pos[None] = ti.math.vec3(light_x, light_y, light_z)

            # Compute elevation and day factor
            sun_dir = light_pos[None].normalized()
            elevation = sun_dir.y
            day_factor = smoothstep(-0.15, 0.25, elevation)

            # Sun, moon, sunset colors
            sun_color    = ti.math.vec3(1.00, 0.95, 0.85) * 1.2
            moon_color   = ti.math.vec3(0.65, 0.75, 1.00) * 0.8
            sunset_color = ti.math.vec3(1.00, 0.50, 0.20) * 5.0

            t_trans = max(0.0, min(1.0, (day_factor - 0.2) / 0.6))
            light_color_rgb = sun_color * t_trans + moon_color * (1.0 - t_trans)

            sunset_strength = max(0.0, 1.0 - abs(elevation) / 0.2) * (1.0 - day_factor)
            light_color_rgb = light_color_rgb * (1.0 - sunset_strength * 0.5) + sunset_color * (sunset_strength * 0.5)

            # Ambient light
            ambient_day   = ti.math.vec3(0.15, 0.15, 0.20)
            ambient_night = ti.math.vec3(0.02, 0.03, 0.1)
            ambient_color = ambient_day * day_factor + ambient_night * (1.0 - day_factor)

            # ===== Sun / Moon spheres =====
            # Far positions (radius 20 units from scene center)
            sun_center = sun_dir * 20.0
            moon_center = -sun_dir * 20.0

            sun_visible = elevation > 0.0
            moon_visible = not sun_visible

            # Sun size (larger, more prominent)
            sphere_centers[4] = sun_center
            sphere_radii[4] = 0.8 if sun_visible else 0.0

            # Moon size (slightly smaller but still visible)
            sphere_centers[5] = moon_center
            sphere_radii[5] = 0.7 if moon_visible else 0.0

        # ===== Camera control =====
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

        # Render with dynamic sun/moon colors
        render_kernel(pixels,
                      cam_pos, cam_target,
                      FOV, ASPECT_RATIO, MAX_BOUNCES,
                      sphere_centers, sphere_radii, sphere_albedos,
                      sphere_material_types, sphere_ior, num_spheres,
                      ground_y,
                      light_pos[None], light_color_rgb, ambient_color)

        canvas.set_image(pixels)
        window.show()

        if window.is_pressed(ti.ui.ESCAPE):
            break

if __name__ == "__main__":
    main()