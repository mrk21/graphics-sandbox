#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

const int MAX_DEPTH = 7;
const float MAT_DIFFUSE = 0.0;
const float MAT_MIRROR  = 1.0;
const float MAT_GLASS   = 2.0;
const float IOR_GLASS   = 1.5;

struct Vertex {
    vec3 pos;
    vec3 normal;
};

struct ObjDesc {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    vec4 color;  // rgb = base color, a = material type
};

struct RayPayload {
    vec3 color;
    int  depth;
};

layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices  { uvec3  i[]; };

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 3, scalar) readonly buffer ObjDescs { ObjDesc descs[]; };

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadEXT bool isShadowed;

hitAttributeEXT vec2 attribs;

// Schlick's Fresnel approximation
float fresnelSchlick(float cosTheta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// Shade a diffuse surface with Phong + shadow
vec3 shadeDiffuse(vec3 worldPos, vec3 worldNormal, vec3 baseColor) {
    vec3 lightDir = normalize(vec3(1.0, 2.0, 1.0));
    vec3 lightColor = vec3(1.0, 0.95, 0.9);

    // Shadow ray
    isShadowed = true;
    traceRayEXT(tlas,
                gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                0xFF,
                0, 0, 1,
                worldPos + worldNormal * 0.001, 0.001, lightDir, 10000.0,
                1);

    float shadow = isShadowed ? 0.15 : 1.0;

    float ambient = 0.1;
    float diffuse = max(dot(worldNormal, lightDir), 0.0);

    vec3 viewDir = normalize(-gl_WorldRayDirectionEXT);
    vec3 halfDir = normalize(lightDir + viewDir);
    float specular = pow(max(dot(worldNormal, halfDir), 0.0), 32.0) * 0.3;

    vec3 color = baseColor * (ambient + diffuse * shadow * lightColor)
               + specular * shadow * lightColor;

    return color;
}

void main() {
    // Look up geometry data for this instance
    ObjDesc desc = descs[gl_InstanceCustomIndexEXT];
    Vertices vertices = Vertices(desc.vertexAddress);
    Indices  indices  = Indices(desc.indexAddress);

    // Get triangle vertex indices
    uvec3 ind = indices.i[gl_PrimitiveID];
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    // Interpolate normal
    vec3 n0 = vertices.v[ind.x].normal;
    vec3 n1 = vertices.v[ind.y].normal;
    vec3 n2 = vertices.v[ind.z].normal;
    vec3 localNormal = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

    // Transform normal to world space
    vec3 worldNormal = normalize((gl_ObjectToWorldEXT * vec4(localNormal, 0.0)).xyz);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    float matType = desc.color.a;
    vec3 baseColor = desc.color.rgb;
    int currentDepth = payload.depth;

    // ---- DIFFUSE ----
    if (matType < 0.5) {
        vec3 color = shadeDiffuse(worldPos, worldNormal, baseColor);
        // Reinhard tone mapping
        payload.color = color / (color + vec3(1.0));
        return;
    }

    // If we've reached max recursion depth, return a darkened base color
    if (currentDepth >= MAX_DEPTH) {
        payload.color = baseColor * 0.1;
        return;
    }

    vec3 incident = normalize(gl_WorldRayDirectionEXT);

    // ---- MIRROR ----
    if (matType < 1.5) {
        vec3 reflDir = reflect(incident, worldNormal);

        // Fresnel: more reflection at grazing angles
        float cosTheta = max(dot(-incident, worldNormal), 0.0);
        float f0 = 0.04; // base reflectivity for metals approximation
        float fresnel = fresnelSchlick(cosTheta, f0);

        // Trace reflected ray
        payload.depth = currentDepth + 1;
        payload.color = vec3(0.0);
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                    0, 0, 0,
                    worldPos + worldNormal * 0.001, 0.001, reflDir, 10000.0, 0);

        vec3 reflColor = payload.color;

        // Metallic reflection tints with base color
        vec3 color = baseColor * reflColor;

        // Add a small diffuse contribution for realism
        vec3 diffContrib = shadeDiffuse(worldPos, worldNormal, baseColor) * 0.05;
        color += diffContrib;

        // Tone map
        payload.color = color / (color + vec3(1.0));
        return;
    }

    // ---- GLASS (refraction + reflection) ----
    {
        // Determine if we're entering or leaving the glass
        bool entering = dot(incident, worldNormal) < 0.0;
        vec3 faceNormal = entering ? worldNormal : -worldNormal;
        float eta = entering ? (1.0 / IOR_GLASS) : IOR_GLASS;

        float cosI = max(dot(-incident, faceNormal), 0.0);

        // Fresnel (Schlick) for dielectrics
        float f0 = pow((1.0 - IOR_GLASS) / (1.0 + IOR_GLASS), 2.0); // ~0.04 for glass
        float fresnel = fresnelSchlick(cosI, f0);

        // Refraction (Snell's law)
        vec3 refractDir = refract(incident, faceNormal, eta);
        bool totalInternalReflection = (length(refractDir) < 0.001);

        if (totalInternalReflection) {
            fresnel = 1.0;
        }

        vec3 reflDir = reflect(incident, faceNormal);
        vec3 offsetPos = worldPos + faceNormal * 0.002;
        vec3 offsetPosInside = worldPos - faceNormal * 0.002;

        vec3 finalColor = vec3(0.0);

        // Reflection
        payload.depth = currentDepth + 1;
        payload.color = vec3(0.0);
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                    0, 0, 0,
                    offsetPos, 0.001, reflDir, 10000.0, 0);
        vec3 reflColor = payload.color;

        if (!totalInternalReflection) {
            // Refraction
            payload.depth = currentDepth + 1;
            payload.color = vec3(0.0);
            traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, 0xFF,
                        0, 0, 0,
                        offsetPosInside, 0.001, refractDir, 10000.0, 0);
            vec3 refrColor = payload.color;

            // Tint the refracted light with the glass color
            refrColor *= baseColor;

            finalColor = mix(refrColor, reflColor, fresnel);
        } else {
            finalColor = reflColor;
        }

        // No tone mapping for glass - let it preserve HDR from the scene
        payload.color = finalColor;
    }
}
