#version 330 core

in vec2 vUV;
out vec4 fragColor;

uniform float uTime;
uniform vec2 uResolution;
uniform sampler2D uSandTex;
uniform sampler2D uSandNormalMap;
uniform float uCamAngleX;
uniform float uCamAngleY;
uniform float uCamDist;

// ---- Constants ----
const float PI = 3.14159265;
const float IOR = 1.52;
const int MAX_BOUNCES = 6;
const int SAMPLES = 4;  // AA samples per pixel

// ---- Materials ----
const int MAT_NONE  = 0;
const int MAT_FLOOR = 1;
const int MAT_GLASS = 2;
const int MAT_METAL = 3;
const int MAT_LIGHT = 4;
const int MAT_GRASS = 5;
const int MAT_BLACKHOLE = 6;

// ---- Object definitions ----
const vec3 SPHERE_CENTER = vec3(0.0, 1.0, 0.0);
const float SPHERE_RADIUS = 1.0;

const vec3 BLOCK1_MIN = vec3(-3.5, 0.0, -1.0);
const vec3 BLOCK1_MAX = vec3(-2.0, 1.5, 0.5);
const vec3 BLOCK2_MIN = vec3(2.0, 0.0, -2.0);
const vec3 BLOCK2_MAX = vec3(3.0, 1.0, -1.0);

const vec3 GRASS_MIN = vec3(1.5, 0.0, 1.0);
const vec3 GRASS_MAX = vec3(2.5, 1.0, 2.0);

// Black hole (scene-scale, near blocks)
// Using Schwarzschild metric: rs = 2GM/c² (event horizon = Schwarzschild radius)
const vec3 BH_CENTER = vec3(-1.0, 1.8, -3.0);
const float BH_SCHWARZSCHILD_R = 0.24;     // Schwarzschild radius rs = 2GM/c²
const float BH_EVENT_HORIZON = BH_SCHWARZSCHILD_R;  // event horizon = rs
const float BH_PHOTON_SPHERE = 1.5 * BH_SCHWARZSCHILD_R;  // photon sphere at 1.5 rs
const float BH_INFLUENCE_RADIUS = 2.5;     // gravitational influence range
const int BH_MARCH_STEPS = 128;            // ray march steps for lensing
const float BH_STEP_SIZE = 0.04;           // step size for ray marching


// Spotlight
const vec3 LIGHT_POS = vec3(1.5, 3.5, 1.5);
const float LIGHT_RADIUS = 0.15;
const vec3 LIGHT_COLOR = vec3(1.0, 0.92, 0.75) * 12.0;
const vec3 LIGHT_DIR = normalize(vec3(0.3, -1.0, 0.1));  // angled toward sand block area
const float LIGHT_INNER_CONE = 0.88;  // cos(~28 deg) - full intensity
const float LIGHT_OUTER_CONE = 0.70;  // cos(~46 deg) - wider falloff edge

// ---- Random / hash ----
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 hash2(vec2 p) {
    return vec2(hash(p), hash(p + vec2(127.1, 311.7)));
}

// ---- Ray / Hit ----
struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 pos;
    vec3 normal;
    int material;
    bool inside;
};

// ---- SDF primitives ----
float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float smax(float a, float b, float k) {
    return -smin(-a, -b, k);
}

// ---- Stanford Bunny SDF (approximation) ----
const vec3 BUNNY_POS = vec3(3.0, 0.0, 1.0);
const float BUNNY_SCALE = 0.55;

float sdBunny(vec3 p) {
    // Transform to local space
    vec3 q = (p - BUNNY_POS) / BUNNY_SCALE;
    q.y -= 0.9;  // lift center

    float d = 1e10;

    // Body - main torso
    d = sdEllipsoid(q - vec3(0.0, 0.0, 0.0), vec3(0.55, 0.5, 0.45));

    // Rump (back/bottom)
    d = smin(d, sdEllipsoid(q - vec3(0.0, -0.15, -0.25), vec3(0.5, 0.45, 0.5)), 0.2);

    // Head
    float head = sdEllipsoid(q - vec3(0.0, 0.45, 0.35), vec3(0.35, 0.32, 0.3));
    d = smin(d, head, 0.15);

    // Snout
    d = smin(d, sdEllipsoid(q - vec3(0.0, 0.35, 0.6), vec3(0.18, 0.15, 0.18)), 0.1);

    // Left ear
    vec3 earL = q - vec3(-0.1, 0.95, 0.2);
    earL = vec3(earL.x * 0.95 + earL.z * 0.3, earL.y, -earL.x * 0.3 + earL.z * 0.95); // slight rotation
    d = smin(d, sdEllipsoid(earL, vec3(0.06, 0.35, 0.04)), 0.08);

    // Right ear
    vec3 earR = q - vec3(0.1, 0.95, 0.2);
    earR = vec3(earR.x * 0.95 - earR.z * 0.3, earR.y, earR.x * 0.3 + earR.z * 0.95);
    d = smin(d, sdEllipsoid(earR, vec3(0.06, 0.35, 0.04)), 0.08);

    // Tail
    d = smin(d, sdSphere(q - vec3(0.0, -0.05, -0.7), 0.15), 0.1);

    // Front legs
    d = smin(d, sdCapsule(q, vec3(-0.18, -0.3, 0.2), vec3(-0.18, -0.85, 0.25), 0.08), 0.1);
    d = smin(d, sdCapsule(q, vec3(0.18, -0.3, 0.2), vec3(0.18, -0.85, 0.25), 0.08), 0.1);

    // Hind legs (thicker)
    d = smin(d, sdEllipsoid(q - vec3(-0.22, -0.45, -0.3), vec3(0.14, 0.35, 0.16)), 0.12);
    d = smin(d, sdEllipsoid(q - vec3(0.22, -0.45, -0.3), vec3(0.14, 0.35, 0.16)), 0.12);

    // Feet
    d = smin(d, sdEllipsoid(q - vec3(-0.22, -0.8, -0.22), vec3(0.1, 0.06, 0.16)), 0.05);
    d = smin(d, sdEllipsoid(q - vec3(0.22, -0.8, -0.22), vec3(0.1, 0.06, 0.16)), 0.05);
    d = smin(d, sdEllipsoid(q - vec3(-0.18, -0.85, 0.28), vec3(0.08, 0.05, 0.12)), 0.05);
    d = smin(d, sdEllipsoid(q - vec3(0.18, -0.85, 0.28), vec3(0.08, 0.05, 0.12)), 0.05);

    // Flatten bottom so it sits on the floor
    d = smax(d, -(q.y + 0.88), 0.02);

    return d * BUNNY_SCALE;
}

// SDF sphere marching intersection
bool intersectSDF(Ray ray, out float t, out vec3 normal) {
    // Bounding sphere check — skip if ray doesn't pass near bunny
    vec3 bc = BUNNY_POS + vec3(0.0, 0.5, 0.0);  // bunny center
    float br = 0.85;  // bounding radius
    vec3 oc = ray.origin - bc;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - br * br;
    float disc = b * b - c;
    if (disc < 0.0) return false;
    float tStart = max(-b - sqrt(disc), 0.001);
    float tEnd = -b + sqrt(disc);
    if (tEnd < 0.001) return false;

    t = tStart;
    for (int i = 0; i < 96; i++) {
        vec3 p = ray.origin + ray.dir * t;
        float d = sdBunny(p);
        if (d < 0.0005) {
            // Compute normal via gradient
            vec2 e = vec2(0.0005, 0.0);
            normal = normalize(vec3(
                sdBunny(p + e.xyy) - sdBunny(p - e.xyy),
                sdBunny(p + e.yxy) - sdBunny(p - e.yxy),
                sdBunny(p + e.yyx) - sdBunny(p - e.yyx)
            ));
            return true;
        }
        t += d;
        if (t > tEnd) return false;
    }
    return false;
}

// SDF inside check (for refraction)
bool isInsideBunny(vec3 p) {
    return sdBunny(p) < 0.0;
}

// ---- Sphere intersection ----
bool intersectSphere(Ray ray, vec3 center, float radius, out float t1, out float t2) {
    vec3 oc = ray.origin - center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return false;
    float sq = sqrt(disc);
    t1 = -b - sq;
    t2 = -b + sq;
    return true;
}

// ---- AABB intersection ----
bool intersectBox(Ray ray, vec3 bmin, vec3 bmax, out float tNear, out vec3 normal) {
    vec3 invDir = 1.0 / ray.dir;
    vec3 t1 = (bmin - ray.origin) * invDir;
    vec3 t2 = (bmax - ray.origin) * invDir;
    vec3 tMin = min(t1, t2);
    vec3 tMax = max(t1, t2);
    tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar = min(min(tMax.x, tMax.y), tMax.z);
    if (tNear > tFar || tFar < 0.001) return false;
    if (tNear < 0.001) tNear = tFar;

    vec3 hitP = ray.origin + ray.dir * tNear;
    vec3 center = (bmin + bmax) * 0.5;
    vec3 halfSize = (bmax - bmin) * 0.5;
    vec3 d = (hitP - center) / halfSize;
    normal = vec3(0.0);
    if (abs(d.x) > abs(d.y) && abs(d.x) > abs(d.z))
        normal.x = sign(d.x);
    else if (abs(d.y) > abs(d.z))
        normal.y = sign(d.y);
    else
        normal.z = sign(d.z);
    return true;
}

// ---- Plane intersection ----
bool intersectPlane(Ray ray, out float t) {
    if (abs(ray.dir.y) < 1e-6) return false;
    t = -ray.origin.y / ray.dir.y;
    return t > 0.001;
}

// ---- Scene intersection ----
Hit sceneIntersect(Ray ray) {
    Hit best;
    best.t = 1e20;
    best.material = MAT_NONE;
    best.inside = false;

    float tFloor;
    if (intersectPlane(ray, tFloor) && tFloor < best.t) {
        best.t = tFloor;
        best.pos = ray.origin + ray.dir * tFloor;
        best.normal = vec3(0.0, 1.0, 0.0);
        best.material = MAT_FLOOR;
    }

    float tS1, tS2;
    if (intersectSphere(ray, SPHERE_CENTER, SPHERE_RADIUS, tS1, tS2)) {
        if (tS1 > 0.001 && tS1 < best.t) {
            best.t = tS1;
            best.pos = ray.origin + ray.dir * tS1;
            best.normal = normalize(best.pos - SPHERE_CENTER);
            best.material = MAT_GLASS;
            best.inside = false;
        } else if (tS2 > 0.001 && tS2 < best.t) {
            best.t = tS2;
            best.pos = ray.origin + ray.dir * tS2;
            best.normal = -normalize(best.pos - SPHERE_CENTER);
            best.material = MAT_GLASS;
            best.inside = true;
        }
    }

    float tB; vec3 nB;
    if (intersectBox(ray, BLOCK1_MIN, BLOCK1_MAX, tB, nB) && tB > 0.001 && tB < best.t) {
        best.t = tB; best.pos = ray.origin + ray.dir * tB;
        best.normal = nB; best.material = MAT_METAL;
    }
    if (intersectBox(ray, BLOCK2_MIN, BLOCK2_MAX, tB, nB) && tB > 0.001 && tB < best.t) {
        best.t = tB; best.pos = ray.origin + ray.dir * tB;
        best.normal = nB; best.material = MAT_METAL;
    }
    if (intersectBox(ray, GRASS_MIN, GRASS_MAX, tB, nB) && tB > 0.001 && tB < best.t) {
        best.t = tB; best.pos = ray.origin + ray.dir * tB;
        best.normal = nB; best.material = MAT_GRASS;
    }

    float tL1, tL2;
    if (intersectSphere(ray, LIGHT_POS, LIGHT_RADIUS, tL1, tL2)) {
        float tL = tL1 > 0.001 ? tL1 : tL2;
        if (tL > 0.001 && tL < best.t) {
            best.t = tL; best.pos = ray.origin + ray.dir * tL;
            best.normal = normalize(best.pos - LIGHT_POS);
            best.material = MAT_LIGHT;
        }
    }

    // Glass bunny (SDF)
    float tBunny; vec3 nBunny;
    if (intersectSDF(ray, tBunny, nBunny) && tBunny > 0.001 && tBunny < best.t) {
        best.t = tBunny;
        best.pos = ray.origin + ray.dir * tBunny;
        best.normal = nBunny;
        best.material = MAT_GLASS;
        best.inside = isInsideBunny(ray.origin);
        if (best.inside) best.normal = -best.normal;
    }

    return best;
}

// ---- Fresnel (Schlick) ----
float fresnel(float cosTheta, float ior1, float ior2) {
    float r0 = (ior1 - ior2) / (ior1 + ior2);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0);
}

// ---- Noise functions ----
vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return fract(sin(p) * 43758.5453) * 2.0 - 1.0;
}

float gradientNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);

    return mix(mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                       dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                   mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                       dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
               mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                       dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                   mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                       dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
}

float fbm(vec3 p, int octaves) {
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        if (i >= octaves) break;
        value += amp * gradientNoise(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// ---- Surface UV for box faces ----
vec2 boxUV(vec3 pos, vec3 normal) {
    if (abs(normal.x) > 0.5) return pos.yz;
    if (abs(normal.y) > 0.5) return pos.xz;
    return pos.xy;
}

// ---- Bump normal from noise ----
vec3 bumpNormal(vec3 pos, vec3 normal, float strength, float scale) {
    float eps = 0.002;
    float center = fbm(pos * scale, 5);
    float dx = fbm((pos + vec3(eps, 0, 0)) * scale, 5) - center;
    float dy = fbm((pos + vec3(0, eps, 0)) * scale, 5) - center;
    float dz = fbm((pos + vec3(0, 0, eps)) * scale, 5) - center;
    vec3 grad = vec3(dx, dy, dz) / eps;
    // Project gradient onto surface tangent plane
    grad -= normal * dot(grad, normal);
    return normalize(normal - grad * strength);
}

// ---- Floor texture ----
vec3 floorColor(vec3 pos, out vec3 bumpedNormal) {
    // Tile pattern with grout
    vec2 tileUV = pos.xz;
    vec2 tileID = floor(tileUV);
    vec2 tileF = fract(tileUV);

    // Grout lines
    float grout = 0.03;
    float groutMask = step(grout, tileF.x) * step(grout, tileF.y)
                    * (1.0 - step(1.0 - grout, tileF.x)) * (1.0 - step(1.0 - grout, tileF.y));
    // Invert: 1 = tile, 0 = grout
    groutMask = step(grout, tileF.x) * step(grout, tileF.y)
              * step(tileF.x, 1.0 - grout) * step(tileF.y, 1.0 - grout);

    // Checkerboard base
    float checker = mod(tileID.x + tileID.y, 2.0);

    // Per-tile color variation via noise
    float tileVar = gradientNoise(vec3(tileID * 3.7, 0.0)) * 0.08;

    vec3 col1 = vec3(0.78, 0.76, 0.72) + tileVar;  // light marble
    vec3 col2 = vec3(0.18, 0.18, 0.22) + tileVar;   // dark marble
    vec3 groutCol = vec3(0.35, 0.33, 0.30);

    // Marble veins
    vec3 tilePos = vec3(pos.xz * 4.0, 0.0);
    float veins = fbm(tilePos + vec3(tileID * 5.0, 0.0), 5);
    veins = smoothstep(-0.1, 0.3, veins) * 0.15;

    vec3 tileCol = mix(col2, col1, checker);
    tileCol -= veins * vec3(0.1, 0.08, 0.06);

    vec3 col = mix(groutCol, tileCol, groutMask);

    // Bump
    bumpedNormal = bumpNormal(pos, vec3(0, 1, 0), 0.15, 8.0);

    return col;
}

// ---- Gold metal texture (brushed) ----
struct MetalSurface {
    vec3 color;
    vec3 normal;
    float roughness;
};

MetalSurface goldTexture(vec3 pos, vec3 normal) {
    MetalSurface s;
    vec2 uv = boxUV(pos, normal);

    // Polished gold base
    s.color = vec3(1.0, 0.78, 0.34);
    // Subtle color depth
    float variation = gradientNoise(vec3(uv * 3.0, 0.5)) * 0.02;
    s.color += vec3(variation, variation * 0.5, 0.0);

    // Minimal bump for slight surface imperfection
    s.normal = bumpNormal(pos, normal, 0.02, 12.0);

    s.roughness = 0.03;
    return s;
}

// ---- Polished steel ----
MetalSurface steelTexture(vec3 pos, vec3 normal) {
    MetalSurface s;
    vec2 uv = boxUV(pos, normal);

    // Mirror-like steel
    s.color = vec3(0.92, 0.92, 0.94);
    // Faint grain
    float grain = gradientNoise(vec3(uv * 8.0, 1.0)) * 0.015;
    s.color += grain;

    // Minimal bump
    s.normal = bumpNormal(pos, normal, 0.02, 10.0);

    s.roughness = 0.04;
    return s;
}

// ---- Minecraft grass block texture (image-based) ----
struct GrassSurface {
    vec3 color;
    vec3 normal;
    float roughness;
};

// Compute UV for grass block face from world position and normal
vec2 grassFaceUV(vec3 pos, vec3 normal) {
    vec3 localPos = (pos - GRASS_MIN) / (GRASS_MAX - GRASS_MIN);
    if (abs(normal.y) > 0.5) {
        // Top / bottom face
        return localPos.xz;
    } else if (abs(normal.x) > 0.5) {
        // Left / right face: Z horizontal, Y vertical
        return vec2(localPos.z, 1.0 - localPos.y);
    } else {
        // Front / back face: X horizontal, Y vertical
        return vec2(localPos.x, 1.0 - localPos.y);
    }
}

// Build TBN matrix for an axis-aligned box face
mat3 faceTBN(vec3 normal) {
    vec3 T;
    if (abs(normal.y) > 0.5)
        T = vec3(1.0, 0.0, 0.0);
    else if (abs(normal.x) > 0.5)
        T = vec3(0.0, 0.0, 1.0);
    else
        T = vec3(1.0, 0.0, 0.0);
    vec3 B = cross(normal, T);
    return mat3(T, B, normal);
}

GrassSurface grassBlockTexture(vec3 pos, vec3 normal) {
    GrassSurface s;
    vec2 uv = grassFaceUV(pos, normal);
    // Sample albedo
    s.color = texture(uSandTex, uv).rgb;
    // Convert from sRGB to linear for correct lighting
    s.color = pow(s.color, vec3(2.2));
    // Sample normal map and transform from tangent space to world space
    vec3 normalTS = texture(uSandNormalMap, uv).rgb * 2.0 - 1.0;
    mat3 tbn = faceTBN(normal);
    s.normal = normalize(tbn * normalTS);
    s.roughness = 0.90;
    return s;
}

// ---- Get metal surface properties ----
MetalSurface getMetalSurface(vec3 pos, vec3 normal) {
    if (pos.x < 0.0) return goldTexture(pos, normal);
    return steelTexture(pos, normal);
}

// ---- Procedural sky (dim, evening/indoor mood) ----
vec3 envColor(vec3 dir) {
    float h = dir.y;
    vec3 horizon = vec3(0.03, 0.03, 0.05);
    vec3 zenith = vec3(0.005, 0.005, 0.02);
    vec3 ground = vec3(0.03, 0.025, 0.02);

    vec3 sky;
    if (h > 0.0) {
        float t = pow(h, 0.5);
        sky = mix(horizon, zenith, t);
    } else {
        sky = mix(horizon, ground, min(-h * 4.0, 1.0));
    }

    // Dim moon/sun glow
    vec3 sunDir = normalize(vec3(2.0, 4.0, -1.0));
    float sunDot = max(dot(dir, sunDir), 0.0);
    sky += vec3(0.6, 0.65, 0.7) * pow(sunDot, 256.0) * 1.5;
    sky += vec3(0.4, 0.42, 0.5) * pow(sunDot, 32.0) * 0.08;

    // Milky Way band — rotated coordinate for band orientation
    vec3 mwDir = vec3(dir.x * 0.7 + dir.z * 0.7, dir.y, -dir.x * 0.7 + dir.z * 0.7);
    float bandDist = abs(mwDir.x + mwDir.y * 0.3);  // tilted band
    float bandShape = exp(-bandDist * bandDist * 6.0);
    // Core is brighter and narrower
    float bandCore = exp(-bandDist * bandDist * 25.0);

    // Large-scale cloud structure
    float cloud1 = fbm(dir * 5.0 + vec3(0.0, 0.0, 1.5), 5) * 0.5 + 0.5;
    float cloud2 = fbm(dir * 10.0 + vec3(3.7, 1.2, 0.0), 4) * 0.5 + 0.5;
    // Dark dust lanes — cuts through the band
    float dust = fbm(dir * 12.0 + vec3(7.0, 2.0, 5.0), 5);
    float dustMask = smoothstep(-0.1, 0.3, dust);

    // Combine Milky Way layers
    float mwBright = bandShape * cloud1 * dustMask;
    float mwCore = bandCore * cloud2 * dustMask;
    // Color: warm core, cooler edges
    vec3 mwColor = mix(vec3(0.08, 0.07, 0.12), vec3(0.14, 0.12, 0.08), mwCore);
    sky += mwColor * mwBright * step(0.0, dir.y);
    sky += vec3(0.10, 0.08, 0.06) * mwCore * step(0.0, dir.y);

    // Fine nebula detail within the band
    if (bandShape > 0.1) {
        float nebula = fbm(dir * 30.0, 4) * 0.5 + 0.5;
        float nebulaDetail = fbm(dir * 60.0, 3) * 0.5 + 0.5;
        sky += vec3(0.05, 0.03, 0.07) * nebula * nebulaDetail * bandShape * dustMask * step(0.0, dir.y);
    }

    // Stars — use 2D projection to avoid grid artifacts
    vec3 starUp = vec3(0.0, 1.0, 0.0);
    vec3 starRight = normalize(cross(dir, starUp));
    if (length(starRight) < 0.01) starRight = vec3(1.0, 0.0, 0.0);

    for (int layer = 0; layer < 3; layer++) {
        float cellSize = 120.0 + float(layer) * 60.0;
        float threshold = 0.992 - float(layer) * 0.004;
        float layerSeed = float(layer) * 53.7;

        // Spherical coordinates for uniform distribution
        float phi = atan(dir.z, dir.x);
        float theta = acos(clamp(dir.y, -1.0, 1.0));
        vec2 starUV = vec2(phi * (0.5 / PI) + 0.5, theta / PI) * cellSize;
        vec2 cellID = floor(starUV);
        vec2 cellF = fract(starUV);

        // Check center + 4 neighbors to avoid edge clipping
        for (int ox = -1; ox <= 1; ox++) {
            for (int oy = -1; oy <= 1; oy++) {
                vec2 neighbor = vec2(float(ox), float(oy));
                vec2 nID = cellID + neighbor;
                float h = hash(nID + layerSeed);
                if (h < threshold) continue;

                // Random position within cell
                vec2 starPos = hash2(nID + layerSeed + 7.0) ;
                vec2 diff = neighbor + starPos - cellF;
                float distToStar = length(diff);

                // Very sharp point spread
                float core = exp(-distToStar * distToStar * 800.0);
                float halo = exp(-distToStar * 12.0) * 0.08;
                float starBright = (core + halo) * ((h - threshold) / (1.0 - threshold));

                // Color from spectral class
                float temp = hash(nID + layerSeed + 200.0);
                vec3 starCol;
                if (temp < 0.1)       starCol = vec3(1.0, 0.65, 0.35);  // K/M
                else if (temp < 0.3)  starCol = vec3(1.0, 0.93, 0.75);  // G
                else if (temp < 0.65) starCol = vec3(0.95, 0.95, 1.0);  // F/A
                else                  starCol = vec3(0.7, 0.8, 1.0);    // B/O

                // Twinkling
                float twinkle = sin(uTime * (1.5 + h * 4.0) + h * 80.0) * 0.2 + 0.8;

                // Brighter in Milky Way region
                float mwBoost = 1.0 + bandShape * 1.5;

                sky += starCol * starBright * twinkle * mwBoost * (2.5 - float(layer) * 0.4);
            }
        }
    }

    return sky;
}

// ---- Soft shadow (area light sampling) ----
float softShadow(vec3 pos, vec3 normal, vec3 lightCenter, float lightRadius, vec2 seed) {
    float shadow = 0.0;
    const int SHADOW_SAMPLES = 4;
    for (int i = 0; i < SHADOW_SAMPLES; i++) {
        vec2 rnd = hash2(seed + vec2(float(i) * 13.7, float(i) * 7.3));
        // Random point on light sphere
        float theta = rnd.x * 2.0 * PI;
        float phi = acos(1.0 - 2.0 * rnd.y);
        vec3 offset = vec3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)) * lightRadius;
        vec3 samplePos = lightCenter + offset;

        vec3 toLight = samplePos - pos;
        float dist = length(toLight);
        vec3 lightDir = toLight / dist;

        Ray shadowRay;
        shadowRay.origin = pos + normal * 0.005;
        shadowRay.dir = lightDir;
        Hit sh = sceneIntersect(shadowRay);
        if (sh.material == MAT_NONE || sh.material == MAT_LIGHT || sh.t > dist) {
            shadow += 1.0;
        }
    }
    return shadow / float(SHADOW_SAMPLES);
}

// ---- Sun shadow (soft via jitter) ----
float sunShadow(vec3 pos, vec3 normal, vec2 seed) {
    vec3 sunDir = normalize(vec3(2.0, 4.0, -1.0));
    float shadow = 0.0;
    const int SUN_SAMPLES = 2;
    for (int i = 0; i < SUN_SAMPLES; i++) {
        vec2 rnd = hash2(seed + vec2(float(i) * 31.1, float(i) * 17.3)) - 0.5;
        // Slight jitter for soft sun shadow
        vec3 jitter = vec3(rnd.x, 0.0, rnd.y) * 0.03;
        vec3 dir = normalize(sunDir + jitter);

        Ray shadowRay;
        shadowRay.origin = pos + normal * 0.005;
        shadowRay.dir = dir;
        Hit sh = sceneIntersect(shadowRay);
        if (sh.material == MAT_NONE || sh.material == MAT_LIGHT) {
            shadow += 1.0;
        } else if (sh.material == MAT_GLASS) {
            shadow += 0.6;  // caustic approximation: glass lets some light through
        }
    }
    return shadow / float(SUN_SAMPLES);
}

// ---- Shade opaque surface with physically-based lighting ----
vec3 shadeSurface(vec3 pos, vec3 normal, vec3 viewDir, vec3 baseColor, float roughness, vec2 seed) {
    vec3 color = vec3(0.0);
    float r2 = roughness * roughness;
    float specPow = 2.0 / (r2 * r2) - 2.0;  // roughness to specular power

    // Dim ambient sun (weak fill light)
    vec3 sunDir = normalize(vec3(2.0, 4.0, -1.0));
    vec3 sunCol = vec3(0.25, 0.24, 0.28);
    float sunDiff = max(dot(normal, sunDir), 0.0);
    float sunVis = sunShadow(pos, normal, seed);
    color += sunCol * sunDiff * sunVis * baseColor;

    // Spotlight
    vec3 toLight = LIGHT_POS - pos;
    float dist = length(toLight);
    vec3 lightDir = toLight / dist;
    float atten = 1.0 / (1.0 + 0.08 * dist * dist);
    // Cone attenuation: how aligned is -lightDir with LIGHT_DIR
    float spotCos = dot(-lightDir, LIGHT_DIR);
    float spotAtten = smoothstep(LIGHT_OUTER_CONE, LIGHT_INNER_CONE, spotCos);
    float diff = max(dot(normal, lightDir), 0.0);
    float lightVis = softShadow(pos, normal, LIGHT_POS, LIGHT_RADIUS, seed + vec2(73.1));
    vec3 H = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, H), 0.0), specPow);
    float fr = fresnel(max(dot(viewDir, H), 0.0), 1.0, 1.5);
    vec3 lCol = LIGHT_COLOR * atten * spotAtten;
    color += lCol * diff * lightVis * baseColor;
    color += lCol * spec * lightVis * mix(0.04, 1.0, fr) * 0.3;

    // Dim ambient
    float skyAmount = 0.5 + 0.5 * normal.y;
    color += baseColor * vec3(0.03, 0.03, 0.05) * skyAmount;
    float groundAmount = 0.5 - 0.5 * normal.y;
    color += baseColor * vec3(0.015, 0.012, 0.01) * groundAmount;

    return color;
}

// ---- Gravitational lensing ray march ----
// Returns true if ray was absorbed by event horizon
// Modifies ray origin and direction via gravitational deflection
struct BHResult {
    bool absorbed;
    bool hitScene;  // hit a scene object during lensing march
    float edgeFade; // 0 at capture radius, 1 far from BH
    vec3 color;     // accumulated emission
    vec3 volumetric; // volumetric light accumulated along curved path
    Ray ray;        // deflected ray
};

BHResult traceBlackHole(Ray ray) {
    BHResult result;
    result.absorbed = false;
    result.hitScene = false;
    result.edgeFade = 1.0;
    result.color = vec3(0.0);
    result.volumetric = vec3(0.0);
    result.ray = ray;

    // Quick check: does ray pass near the black hole?
    vec3 oc = ray.origin - BH_CENTER;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - BH_INFLUENCE_RADIUS * BH_INFLUENCE_RADIUS;
    float disc = b * b - c;
    if (disc < 0.0) return result;  // ray doesn't enter influence zone

    // Advance ray to influence sphere entry point
    float sqrtDisc = sqrt(disc);
    float tEntry = -b - sqrtDisc;
    vec3 pos = ray.origin;
    vec3 dir = ray.dir;
    if (tEntry > 0.0) {
        // Ray starts outside: jump to entry point
        pos += dir * (tEntry - 0.01);
    }
    float totalDist = 0.0;
    float minDist = 1e20;

    for (int i = 0; i < BH_MARCH_STEPS; i++) {
        vec3 toCenter = BH_CENTER - pos;
        float dist = length(toCenter);

        // Photon capture radius (~2.6 rs for Schwarzschild)
        float captureR = BH_SCHWARZSCHILD_R * 2.6;
        if (dist < captureR) {
            result.absorbed = true;
            return result;
        }
        // Track minimum distance for edge fade calculation
        minDist = min(minDist, dist);

        // Outside influence: stop marching
        if (dist > BH_INFLUENCE_RADIUS && totalDist > 0.1) break;

        // GR geodesic acceleration for Schwarzschild metric
        // Deflection is 2x Newtonian, plus enhanced near photon sphere
        float rs = BH_SCHWARZSCHILD_R;
        float grFactor = 1.5 * rs / (dist * dist);
        // Near photon sphere (r = 1.5 rs), light orbits — enhance deflection
        float photonBoost = 1.0 + 2.0 * exp(-(dist - BH_PHOTON_SPHERE) * (dist - BH_PHOTON_SPHERE) / (rs * rs * 0.5));
        // Smooth fadeout near influence boundary (no sharp cutoff)
        float edgeFade = smoothstep(BH_INFLUENCE_RADIUS, BH_INFLUENCE_RADIUS * 0.5, dist);
        vec3 gravity = toCenter / dist * grFactor * photonBoost * edgeFade;

        // Adaptive step size: much smaller steps near event horizon
        float distRatio = dist / BH_EVENT_HORIZON;
        float adaptiveStep = BH_STEP_SIZE * clamp(distRatio * distRatio * 0.3, 0.05, 3.0);

        // Check for scene intersection along deflected ray
        if (i % 4 == 0) {
            Ray testRay;
            testRay.origin = pos;
            testRay.dir = dir;
            Hit testHit = sceneIntersect(testRay);
            if (testHit.material != MAT_NONE && testHit.t < adaptiveStep * 6.0) {
                result.hitScene = true;
                result.ray.origin = pos;
                result.ray.dir = dir;
                return result;
            }
        }

        // Accumulate volumetric light along curved path
        {
            vec3 toLight = LIGHT_POS - pos;
            float lDist = length(toLight);
            vec3 lDir = toLight / lDist;
            float spotCos = dot(-lDir, LIGHT_DIR);
            float spotAtten = smoothstep(LIGHT_OUTER_CONE, LIGHT_INNER_CONE, spotCos);
            if (spotAtten > 0.0) {
                float dAtten = 1.0 / (1.0 + 0.1 * lDist * lDist);
                Ray sRay;
                sRay.origin = pos;
                sRay.dir = lDir;
                Hit sHit = sceneIntersect(sRay);
                float sVis = (sHit.material == MAT_NONE || sHit.material == MAT_LIGHT || sHit.t > lDist) ? 1.0 : 0.0;
                if (sHit.material == MAT_GLASS) sVis = 0.5;
                result.volumetric += LIGHT_COLOR * spotAtten * dAtten * sVis * adaptiveStep * 0.012;
            }
        }

        // Update velocity (bend the ray)
        dir = normalize(dir + gravity * adaptiveStep);

        // Advance position
        pos += dir * adaptiveStep;
        totalDist += adaptiveStep;
    }

    result.ray.origin = pos;
    result.ray.dir = dir;
    return result;
}

// ---- Volumetric light along a straight ray segment ----
vec3 segmentVolumetric(vec3 origin, vec3 dir, float tStart, float tEnd, vec2 seed) {
    vec3 accum = vec3(0.0);
    if (tEnd <= tStart) return accum;
    const int STEPS = 16;
    float stepSize = (tEnd - tStart) / float(STEPS);
    float jitter = hash(seed) * stepSize;
    for (int i = 0; i < STEPS; i++) {
        float t = tStart + jitter + float(i) * stepSize;
        if (t > tEnd) break;
        vec3 pos = origin + dir * t;
        vec3 toLight = LIGHT_POS - pos;
        float dist = length(toLight);
        vec3 lightDir = toLight / dist;
        float spotCos = dot(-lightDir, LIGHT_DIR);
        float spotAtten = smoothstep(LIGHT_OUTER_CONE, LIGHT_INNER_CONE, spotCos);
        if (spotAtten > 0.0) {
            float distAtten = 1.0 / (1.0 + 0.1 * dist * dist);
            Ray sRay;
            sRay.origin = pos;
            sRay.dir = lightDir;
            Hit sh = sceneIntersect(sRay);
            float vis = (sh.material == MAT_NONE || sh.material == MAT_LIGHT || sh.t > dist) ? 1.0 : 0.0;
            if (sh.material == MAT_GLASS) vis = 0.5;
            accum += LIGHT_COLOR * spotAtten * distAtten * vis * stepSize * 0.012;
        }
    }
    return accum;
}

// ---- Main trace ----
vec3 trace(Ray ray, vec2 seed) {
    vec3 color = vec3(0.0);
    vec3 throughput = vec3(1.0);

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        // Check if ray enters BH influence zone, and if so, how far
        float bhEntryDist = 1e20;
        {
            vec3 oc = ray.origin - BH_CENTER;
            float b = dot(oc, ray.dir);
            float c = dot(oc, oc) - BH_INFLUENCE_RADIUS * BH_INFLUENCE_RADIUS;
            float disc = b * b - c;
            if (disc >= 0.0) {
                float sqD = sqrt(disc);
                float t1 = -b - sqD;
                float t2 = -b + sqD;
                if (t2 > 0.0) bhEntryDist = max(t1, 0.0);
            }
        }

        Hit hit = sceneIntersect(ray);

        // Apply BH lensing if influence zone is closer than scene hit
        if (bhEntryDist < hit.t) {
            color += throughput * segmentVolumetric(ray.origin, ray.dir, 0.0, bhEntryDist, seed + vec2(53.7));

            BHResult bh = traceBlackHole(ray);
            color += throughput * bh.color;
            color += throughput * bh.volumetric;
            if (bh.absorbed) break;
            ray = bh.ray;
            hit = sceneIntersect(ray);

            float postDist = min(hit.t, 12.0);
            color += throughput * segmentVolumetric(ray.origin, ray.dir, 0.0, postDist, seed + vec2(97.3));
        } else {
            float volDist = min(hit.t, 12.0);
            color += throughput * segmentVolumetric(ray.origin, ray.dir, 0.0, volDist, seed + vec2(53.7));
        }

        if (hit.material == MAT_NONE) {
            color += throughput * envColor(ray.dir);
            break;
        }

        vec3 viewDir = -ray.dir;

        if (hit.material == MAT_LIGHT) {
            // Brighter when viewed from beam direction, dimmer from behind
            float facing = dot(-ray.dir, LIGHT_DIR);
            float glow = smoothstep(-0.2, 1.0, facing);
            // Hot center, warm edge
            vec3 hotColor = vec3(1.0, 0.98, 0.95) * 15.0;
            vec3 warmEdge = vec3(1.0, 0.7, 0.3) * 4.0;
            float edgeFactor = 1.0 - abs(dot(-ray.dir, hit.normal));
            vec3 emissive = mix(hotColor, warmEdge, edgeFactor) * (0.3 + 0.7 * glow);
            color += throughput * emissive;
            break;
        }

        if (hit.material == MAT_FLOOR) {
            vec3 floorNorm;
            vec3 base = floorColor(hit.pos, floorNorm);
            color += throughput * shadeSurface(hit.pos, floorNorm, viewDir, base, 0.85, seed);

            // Subtle floor reflection
            float floorFr = fresnel(dot(viewDir, floorNorm), 1.0, 1.5) * 0.15;
            if (floorFr > 0.01) {
                Ray reflRay;
                reflRay.origin = hit.pos + floorNorm * 0.005;
                reflRay.dir = reflect(-viewDir, floorNorm);
                Hit reflHit = sceneIntersect(reflRay);
                vec3 reflCol;
                if (reflHit.material == MAT_NONE) reflCol = envColor(reflRay.dir);
                else if (reflHit.material == MAT_LIGHT) reflCol = LIGHT_COLOR;
                else reflCol = envColor(reflRay.dir) * 0.5;
                color += throughput * floorFr * reflCol;
            }
            break;
        }

        if (hit.material == MAT_METAL) {
            MetalSurface ms = getMetalSurface(hit.pos, hit.normal);
            color += throughput * shadeSurface(hit.pos, ms.normal, viewDir, ms.color, ms.roughness, seed) * 0.25;
            float fr = fresnel(dot(viewDir, ms.normal), 1.0, 2.5);
            throughput *= mix(ms.color, vec3(1.0), fr);
            ray.origin = hit.pos + ms.normal * 0.005;
            ray.dir = reflect(-viewDir, ms.normal);
            continue;
        }

        if (hit.material == MAT_GRASS) {
            GrassSurface gs = grassBlockTexture(hit.pos, hit.normal);
            color += throughput * shadeSurface(hit.pos, gs.normal, viewDir, gs.color, gs.roughness, seed);
            break;
        }

        if (hit.material == MAT_GLASS) {
            float cosI = dot(viewDir, hit.normal);
            float eta = hit.inside ? IOR : (1.0 / IOR);
            float fr = fresnel(cosI, 1.0, IOR);
            vec3 refracted = refract(-viewDir, hit.normal, eta);

            if (length(refracted) < 0.001) {
                ray.origin = hit.pos + hit.normal * 0.005;
                ray.dir = reflect(-viewDir, hit.normal);
                if (hit.inside) throughput *= vec3(0.95, 0.98, 0.99);
                continue;
            }

            // Reflection contribution
            vec3 reflDir = reflect(-viewDir, hit.normal);
            Ray reflRay;
            reflRay.origin = hit.pos + hit.normal * 0.005;
            reflRay.dir = reflDir;
            Hit reflHit = sceneIntersect(reflRay);
            vec3 reflColor;
            if (reflHit.material == MAT_NONE)
                reflColor = envColor(reflDir);
            else if (reflHit.material == MAT_LIGHT)
                reflColor = LIGHT_COLOR;
            else if (reflHit.material == MAT_FLOOR) {
                vec3 fn;
                reflColor = shadeSurface(reflHit.pos, reflHit.normal, -reflDir, floorColor(reflHit.pos, fn), 0.85, seed);
            } else if (reflHit.material == MAT_METAL) {
                MetalSurface rms = getMetalSurface(reflHit.pos, reflHit.normal);
                reflColor = shadeSurface(reflHit.pos, rms.normal, -reflDir, rms.color, rms.roughness, seed);
            } else if (reflHit.material == MAT_GRASS) {
                GrassSurface rgs = grassBlockTexture(reflHit.pos, reflHit.normal);
                reflColor = shadeSurface(reflHit.pos, rgs.normal, -reflDir, rgs.color, rgs.roughness, seed);
            } else
                reflColor = envColor(reflDir);
            color += throughput * fr * reflColor;

            // Refraction
            throughput *= (1.0 - fr);
            if (hit.inside) {
                throughput *= exp(-vec3(0.15, 0.04, 0.04) * hit.t);
            }
            ray.origin = hit.pos - hit.normal * 0.005;
            ray.dir = normalize(refracted);
            continue;
        }
    }

    return color;
}

// ---- Camera ----
Ray getCameraRay(vec2 uv, vec2 jitter) {
    float camHeight = sin(uCamAngleY) * uCamDist;
    float camHorizDist = cos(uCamAngleY) * uCamDist;
    vec3 camPos = vec3(sin(uCamAngleX) * camHorizDist, 1.0 + camHeight, cos(uCamAngleX) * camHorizDist);
    vec3 target = vec3(0.0, 0.8, 0.0);

    vec3 forward = normalize(target - camPos);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);

    // Apply sub-pixel jitter for AA
    vec2 jitteredUV = uv + jitter / uResolution.y;

    float fov = 0.7;
    vec3 dir = normalize(forward + (jitteredUV.x * right + jitteredUV.y * up) * fov);

    Ray ray;
    ray.origin = camPos;
    ray.dir = dir;
    return ray;
}

// ---- Volumetric spotlight (ray marching) ----
vec3 volumetricLight(Ray ray, float maxDist, vec2 seed) {
    const int VOL_STEPS = 32;
    float stepSize = min(maxDist, 12.0) / float(VOL_STEPS);
    vec3 accumLight = vec3(0.0);
    float jitter = hash(seed) * stepSize;  // Jitter start to reduce banding

    // Camera ray's closest approach to BH center
    vec3 camToBH = BH_CENTER - ray.origin;
    float tClosest = dot(camToBH, ray.dir);  // t where ray is closest to BH
    float closestDist2 = dot(camToBH, camToBH) - tClosest * tClosest;  // squared closest distance
    float bhViewShadowR = BH_INFLUENCE_RADIUS * 0.6;

    for (int i = 0; i < VOL_STEPS; i++) {
        float t = jitter + float(i) * stepSize;
        vec3 samplePos = ray.origin + ray.dir * t;

        // Check if sample is in spotlight cone
        vec3 toLight = LIGHT_POS - samplePos;
        float dist = length(toLight);
        vec3 lightDir = toLight / dist;
        float spotCos = dot(-lightDir, LIGHT_DIR);
        float spotAtten = smoothstep(LIGHT_OUTER_CONE, LIGHT_INNER_CONE, spotCos);

        if (spotAtten > 0.0) {
            float distAtten = 1.0 / (1.0 + 0.1 * dist * dist);

            // Simple occlusion check (skip full scene trace for perf)
            Ray shadowRay;
            shadowRay.origin = samplePos;
            shadowRay.dir = lightDir;
            Hit sh = sceneIntersect(shadowRay);
            float vis = (sh.material == MAT_NONE || sh.material == MAT_LIGHT || sh.t > dist) ? 1.0 : 0.0;
            // Glass is semi-transparent
            if (sh.material == MAT_GLASS) vis = 0.5;

            // Black hole occlusion: scattered light from behind BH can't reach camera
            // If sample is beyond BH's closest approach on the camera ray, attenuate
            float bhVis = 1.0;
            if (t > tClosest && tClosest > 0.0) {
                bhVis = smoothstep(BH_EVENT_HORIZON * BH_EVENT_HORIZON, bhViewShadowR * bhViewShadowR, closestDist2);
            }

            accumLight += LIGHT_COLOR * spotAtten * distAtten * vis * stepSize * 0.012 * bhVis;
        }
    }
    return accumLight;
}

// ---- ACES tone mapping ----
vec3 acesToneMap(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - uResolution * 0.5) / uResolution.y;

    vec3 color = vec3(0.0);

    for (int s = 0; s < SAMPLES; s++) {
        // Sub-pixel jitter for anti-aliasing
        vec2 seed = gl_FragCoord.xy + vec2(float(s) * 127.1, uTime * 11.3);
        vec2 jitter = (hash2(seed) - 0.5);
        Ray ray = getCameraRay(uv, jitter);
        color += trace(ray, seed);
    }
    color /= float(SAMPLES);

    // ACES tone mapping
    color = acesToneMap(color);

    // Gamma
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}
