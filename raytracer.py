import taichi as ti

EPS = 1e-4

# material types
DIFFUSE = 0
MIRROR = 1
DIELECTRIC = 2
EMISSIVE = 3

@ti.dataclass
class Ray:
    origin: ti.math.vec3
    direction: ti.math.vec3

@ti.dataclass
class HitRecord:
    hit: ti.i32
    point: ti.math.vec3
    normal: ti.math.vec3
    t: ti.f32
    material_type: ti.i32
    albedo: ti.math.vec3
    ior: ti.f32
    sphere_id: ti.i32

# ------------------------------------------------------------------
# Intersection routines
# ------------------------------------------------------------------
@ti.func
def intersect_sphere(ray, center, radius):
    oc = ray.origin - center
    a = ray.direction.dot(ray.direction)
    b = 2.0 * oc.dot(ray.direction)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c

    t = -1.0
    if disc >= 0.0:
        sqrt_d = ti.sqrt(disc)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > EPS:
            t = t1
        elif t2 > EPS:
            t = t2
    return t

@ti.func
def intersect_ground(ray, ground_y):
    t = -1.0
    if abs(ray.direction.y) >= EPS:
        t_candidate = (ground_y - ray.origin.y) / ray.direction.y
        if t_candidate > EPS:
            t = t_candidate
    return t

# ------------------------------------------------------------------
# World traversal
# ------------------------------------------------------------------
@ti.func
def trace_world(ray,
                sphere_centers, sphere_radii, sphere_albedos, sphere_material_types, sphere_ior,
                num_spheres,
                ground_y):
    hit_record = HitRecord()
    hit_record.hit = 0
    hit_record.sphere_id = -1
    closest_t = 1e9

    for i in range(num_spheres):
        t = intersect_sphere(ray, sphere_centers[i], sphere_radii[i])
        if t > 0 and t < closest_t:
            closest_t = t
            hit_record.hit = 1
            hit_record.t = t
            hit_record.point = ray.origin + ray.direction * t
            hit_record.normal = (hit_record.point - sphere_centers[i]).normalized()
            hit_record.material_type = sphere_material_types[i]
            hit_record.albedo = sphere_albedos[i]
            hit_record.ior = sphere_ior[i]
            hit_record.sphere_id = i

    t_g = intersect_ground(ray, ground_y)
    if t_g > 0 and t_g < closest_t:
        closest_t = t_g
        hit_record.hit = 1
        hit_record.t = t_g
        hit_record.point = ray.origin + ray.direction * t_g
        hit_record.normal = ti.math.vec3(0.0, 1.0, 0.0)
        hit_record.material_type = DIFFUSE
        xz = ti.floor(hit_record.point.x * 2.0) + ti.floor(hit_record.point.z * 2.0)
        if ti.abs(xz) % 2 == 0:
            hit_record.albedo = ti.math.vec3(0.3, 0.3, 0.3)
        else:
            hit_record.albedo = ti.math.vec3(0.7, 0.7, 0.7)
        hit_record.ior = 1.0
        hit_record.sphere_id = -1

    return hit_record

# ------------------------------------------------------------------
# Shadow attenuation (fully corrected)
# ------------------------------------------------------------------
@ti.func
def compute_shadow_attenuation(ray, light_dist,
                               sphere_centers, sphere_radii, sphere_albedos, sphere_material_types, sphere_ior,
                               num_spheres, ground_y):
    attenuation = ti.math.vec3(1.0)
    remaining_dist = light_dist
    current_ray = ray
    max_steps = 20

    for _ in range(max_steps):
        hit = trace_world(current_ray,
                          sphere_centers, sphere_radii, sphere_albedos,
                          sphere_material_types, sphere_ior,
                          num_spheres, ground_y)
        if not hit.hit:
            break
        if hit.t > remaining_dist - EPS:
            break

        if hit.material_type == DIFFUSE:
            # Opaque diffuse – completely blocks light
            attenuation = ti.math.vec3(0.0)
            break
        elif hit.material_type == MIRROR:
            # Mirror – reflects almost all light; only a small, neutral absorption.
            # Adjust the constant below (e.g., 0.05 for 95% reflection, 5% absorption).
            # This gives a nearly invisible light‑gray shadow without unnatural colors.
            absorption_strength = 0.05   # tunable: smaller → lighter shadow
            attenuation = ti.math.vec3(absorption_strength)
            break   # shadow ray stops here (mirror does not transmit)
        elif hit.material_type == DIELECTRIC:
            # Glass – compute absorption inside the sphere
            c = sphere_centers[hit.sphere_id]
            r = sphere_radii[hit.sphere_id]
            oc = current_ray.origin - c
            a = current_ray.direction.dot(current_ray.direction)
            b = 2.0 * oc.dot(current_ray.direction)
            c_val = oc.dot(oc) - r * r
            disc = b * b - 4.0 * a * c_val

            if disc >= 0.0:
                sqrt_d = ti.sqrt(disc)
                t1 = (-b - sqrt_d) / (2.0 * a)
                t2 = (-b + sqrt_d) / (2.0 * a)
                t_exit = 0.0
                if t1 > EPS:
                    t_exit = t1
                if t2 > EPS and (t_exit < EPS or t2 < t_exit):
                    t_exit = t2
                dist_inside = max(0.0, t_exit)
                # Lower absorption coefficient for lighter shadows
                absorption_coeff = ti.math.vec3(0.25, 0.25, 0.25)   # tune this
                attenuation *= hit.albedo * ti.exp(-absorption_coeff * dist_inside)
            else:
                attenuation *= hit.albedo

            # Continue shadow ray through the glass
            new_origin = hit.point + current_ray.direction * EPS
            current_ray = Ray(origin=new_origin, direction=current_ray.direction)
            remaining_dist -= hit.t
        else:
            attenuation = ti.math.vec3(0.0)
            break

        if attenuation.max() < 0.01:
            break

    return attenuation

# ------------------------------------------------------------------
# Lighting
# ------------------------------------------------------------------
@ti.func
def calculate_lighting(hit, view_dir,
                       light_pos, light_color, ambient,
                       sphere_centers, sphere_radii, sphere_albedos, sphere_material_types, sphere_ior,
                       num_spheres, ground_y):
    light_dir = (light_pos - hit.point).normalized()
    light_dist = (light_pos - hit.point).norm()
    shadow_ray = Ray(origin=hit.point + hit.normal * EPS, direction=light_dir)
    shadow_atten = compute_shadow_attenuation(shadow_ray, light_dist,
                                              sphere_centers, sphere_radii, sphere_albedos,
                                              sphere_material_types, sphere_ior,
                                              num_spheres, ground_y)

    effective_light_color = light_color * shadow_atten

    color = ambient * hit.albedo
    diff = max(0.0, hit.normal.dot(light_dir))
    intensity = effective_light_color * diff
    reflect_dir = (2.0 * hit.normal.dot(light_dir) * hit.normal - light_dir).normalized()
    spec = max(0.0, reflect_dir.dot(-view_dir)) ** 32
    spec_color = ti.math.vec3(0.5, 0.5, 0.5) * spec * shadow_atten

    color += (intensity + spec_color) * hit.albedo
    return color

# ------------------------------------------------------------------
# Reflection, refraction, Fresnel
# ------------------------------------------------------------------
@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n

@ti.func
def refract(v, n, eta):
    cos_theta = -v.dot(n)
    sin_theta_sq = 1.0 - cos_theta * cos_theta
    sin_phi_sq = eta * eta * sin_theta_sq
    result = ti.math.vec3(0.0)
    if sin_phi_sq <= 1.0:
        cos_phi = ti.sqrt(1.0 - sin_phi_sq)
        result = eta * v + (eta * cos_theta - cos_phi) * n
    return result

@ti.func
def fresnel_schlick(cos_theta, n1, n2):
    r0 = (n1 - n2) / (n1 + n2)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_theta, 5.0)

# ------------------------------------------------------------------
# Full ray tracing
# ------------------------------------------------------------------
@ti.func
def trace_ray(ray, max_depth,
              sphere_centers, sphere_radii, sphere_albedos, sphere_material_types, sphere_ior,
              num_spheres, ground_y, light_pos, light_color, ambient):
    color = ti.math.vec3(0.0)
    throughput = ti.math.vec3(1.0)
    current_ray = ray

    for _ in range(max_depth):
        hit = trace_world(current_ray,
                          sphere_centers, sphere_radii, sphere_albedos,
                          sphere_material_types, sphere_ior,
                          num_spheres, ground_y)

        if not hit.hit:
            # Dynamic sky color based on light position
            sun_dir = light_pos.normalized()
            elevation = sun_dir.y
            day_factor = ti.math.smoothstep(-0.15, 0.25, elevation)

            top_color_day = ti.math.vec3(0.20, 0.40, 0.80)
            bottom_color_day = ti.math.vec3(0.70, 0.80, 1.00)
            top_color_night = ti.math.vec3(0.02, 0.02, 0.05)
            bottom_color_night = ti.math.vec3(0.05, 0.05, 0.10)

            top_color = ti.math.mix(top_color_night, top_color_day, day_factor)
            bottom_color = ti.math.mix(bottom_color_night, bottom_color_day, day_factor)

            sunset_factor = 1.0 - ti.abs(elevation) / 0.25
            sunset_factor = ti.max(0.0, sunset_factor) * (1.0 - day_factor)
            sunset_color = ti.math.vec3(1.0, 0.5, 0.2)
            bottom_color = ti.math.mix(bottom_color, sunset_color, sunset_factor * 0.6)

            t = (current_ray.direction.y + 1.0) * 0.5
            sky_color = ti.math.mix(bottom_color, top_color, t)
            color += throughput * sky_color
            break

        view_dir = -current_ray.direction

        if hit.material_type == DIFFUSE:
            direct = calculate_lighting(hit, view_dir,
                                        light_pos, light_color, ambient,
                                        sphere_centers, sphere_radii, sphere_albedos,
                                        sphere_material_types, sphere_ior,
                                        num_spheres, ground_y)
            color += throughput * direct
            throughput *= hit.albedo * 0.3
            n = hit.normal
            u1 = ti.random()
            u2 = ti.random()
            r = ti.sqrt(u1)
            theta = 2.0 * 3.14159 * u2
            x = r * ti.cos(theta)
            y = r * ti.sin(theta)
            z = ti.sqrt(1.0 - u1)
            bounce_dir = ti.math.vec3(x, y, z)
            if bounce_dir.dot(n) < 0.0:
                bounce_dir = -bounce_dir
            current_ray = Ray(origin=hit.point + n * EPS, direction=bounce_dir)

        elif hit.material_type == MIRROR:
            throughput *= hit.albedo
            n = hit.normal
            bounce_dir = reflect(current_ray.direction, n)
            current_ray = Ray(origin=hit.point + n * EPS, direction=bounce_dir)
        elif hit.material_type == EMISSIVE:
            # Add emissive glow (color = albedo * brightness)
            color += throughput * hit.albedo * 2.0  # brightness factor
            break  # do not continue bouncing
        else:  # DIELECTRIC
            outward_normal = hit.normal
            cos_theta = current_ray.direction.dot(outward_normal)

            normal = outward_normal
            n1 = 1.0
            n2 = hit.ior
            cos_theta_i = -cos_theta

            if cos_theta > 0.0:  # exiting
                normal = -outward_normal
                n1 = hit.ior
                n2 = 1.0
                cos_theta_i = cos_theta

            eta = n1 / n2
            sin_theta_t_sq = eta * eta * (1.0 - cos_theta_i * cos_theta_i)

            bounce_dir = ti.math.vec3(0.0)
            reflect_prob = 0.0
            new_origin = hit.point

            if sin_theta_t_sq > 1.0:  # total internal reflection
                bounce_dir = reflect(current_ray.direction, normal)
                reflect_prob = 1.0
            else:
                bounce_dir_refract = refract(current_ray.direction, normal, eta)
                reflect_prob = fresnel_schlick(cos_theta_i, n1, n2)
                if ti.random() < reflect_prob:
                    bounce_dir = reflect(current_ray.direction, normal)
                else:
                    bounce_dir = bounce_dir_refract

            if sin_theta_t_sq > 1.0 or ti.random() < reflect_prob:
                throughput *= reflect_prob * hit.albedo
                new_origin = hit.point + normal * EPS
            else:
                throughput *= (1.0 - reflect_prob) * hit.albedo
                new_origin = hit.point + bounce_dir * EPS
            current_ray = Ray(origin=new_origin, direction=bounce_dir)

    return color

# ------------------------------------------------------------------
# Main render kernel
# ------------------------------------------------------------------
@ti.kernel
def render_kernel(pixels: ti.template(),
                  cam_pos: ti.math.vec3, cam_target: ti.math.vec3,
                  fov: ti.f32, aspect_ratio: ti.f32, max_bounces: ti.i32,
                  sphere_centers: ti.template(), sphere_radii: ti.template(),
                  sphere_albedos: ti.template(), sphere_material_types: ti.template(),
                  sphere_ior: ti.template(), num_spheres: ti.i32,
                  ground_y: ti.f32,
                  light_pos: ti.math.vec3,
                  light_color: ti.math.vec3,
                  ambient: ti.math.vec3):
    WIDTH = pixels.shape[0]
    HEIGHT = pixels.shape[1]

    forward = (cam_target - cam_pos).normalized()
    right = ti.math.vec3(0.0, 1.0, 0.0).cross(forward).normalized()
    up = forward.cross(right).normalized()

    for i, j in ti.ndrange(WIDTH, HEIGHT):
        u = (2.0 * i / WIDTH - 1.0) * ti.tan(fov / 2.0) * aspect_ratio
        v = (2.0 * j / HEIGHT - 1.0) * ti.tan(fov / 2.0)

        ray_dir = (forward + right * u + up * v).normalized()
        ray = Ray(origin=cam_pos, direction=ray_dir)

        color = trace_ray(ray, max_bounces,
                          sphere_centers, sphere_radii, sphere_albedos,
                          sphere_material_types, sphere_ior,
                          num_spheres, ground_y,
                          light_pos, light_color, ambient)

        color = ti.math.clamp(color, 0.0, 1.0)
        color = ti.math.pow(color, 1.0 / 2.2)
        pixels[i, j] = color