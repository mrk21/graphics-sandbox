#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    int  depth;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    // Sky gradient
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    float t = 0.5 * (dir.y + 1.0);
    vec3 sky = mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);

    // Dim sun glow
    vec3 sunDir = normalize(vec3(1.0, 2.0, 1.0));
    float sunDot = max(dot(dir, sunDir), 0.0);
    sky += vec3(1.0, 0.95, 0.8) * pow(sunDot, 128.0) * 2.0;

    payload.color = sky;
}
