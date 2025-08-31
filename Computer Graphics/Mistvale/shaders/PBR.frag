#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragTan;

layout(location = 0) out vec4 outColor;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;
layout(binding = 2, set = 1) uniform sampler2D normalMap;
layout(binding = 3, set = 1) uniform sampler2D metallicMap;
layout(binding = 4, set = 1) uniform sampler2D roughnessMap;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 lightDir;
    vec4 lightColor;
    vec3 eyePos;
    vec3 blueLightPos;
    vec4 blueLightColor;
} gubo;

const float PI = 3.14159265359;

mat3 computeTBN(vec3 N, vec3 T, float tangentW) {
    vec3 B = cross(N, T) * tangentW;
    return mat3(normalize(T), normalize(B), normalize(N));
}

vec3 getNormalFromMap(mat3 TBN) {
    vec3 tangentNormal = texture(normalMap, fragUV).xyz * 2.0 - 1.0;
    return normalize(TBN * tangentNormal);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0001f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0001f), roughness) *
    GeometrySchlickGGX(max(dot(N, L), 0.0001f), roughness);
}

void main() {
    vec3 albedo     = texture(albedoMap, fragUV).rgb; 
    float metallic  = texture(metallicMap, fragUV).r;
    float roughness = texture(roughnessMap, fragUV).r;


    vec3 N = normalize(fragNorm);
    vec3 T = normalize(fragTan.xyz);
    float w = fragTan.w;
    mat3 TBN = computeTBN(N, T, w);
    vec3 Nmap = getNormalFromMap(TBN);

    vec3 V = normalize(gubo.eyePos - fragPos);
    
    // Main directional light
    vec3 L = normalize(gubo.lightDir);
    vec3 H = normalize(V + L);
    vec3 radiance = gubo.lightColor.rgb;

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float NDF = DistributionGGX(Nmap, H, roughness);
    float G   = GeometrySmith(Nmap, V, L, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) /
    max(4.0 * max(dot(Nmap, V), 0.0001f) * max(dot(Nmap, L), 0.0), 0.0001f);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(Nmap, L), 0.0);
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    // Blue spotlight
    vec3 blueLightDir = normalize(gubo.blueLightPos - fragPos);
    float blueLightDist = length(gubo.blueLightPos - fragPos);
    float blueLightAttenuation = 1.0 / (1.0 + 0.05 * blueLightDist + 0.005 * blueLightDist * blueLightDist);
    
    // Spotlight cone effect - create a narrow beam pointing downward
    vec3 spotlightDir = normalize(vec3(0.0, -1.0, 0.0)); // Pointing straight down
    float cosAngle = dot(-blueLightDir, spotlightDir);
    float spotCutoff = cos(radians(8.0)); // 8 degree cone - very narrow like flashlight
    float spotFalloff = cos(radians(12.0)); // 12 degree falloff
    
    float spotEffect = 0.0;
    if(cosAngle > spotCutoff) {
        spotEffect = pow((cosAngle - spotCutoff) / (spotFalloff - spotCutoff), 2.0);
    }
    
    vec3 blueH = normalize(V + blueLightDir);
    vec3 blueRadiance = gubo.blueLightColor.rgb * blueLightAttenuation * spotEffect * gubo.blueLightColor.a;
    
    float blueNDF = DistributionGGX(Nmap, blueH, roughness);
    float blueG = GeometrySmith(Nmap, V, blueLightDir, roughness);
    vec3 blueF = fresnelSchlick(max(dot(blueH, V), 0.0), F0);
    
    vec3 blueSpecular = (blueNDF * blueG * blueF) /
    max(4.0 * max(dot(Nmap, V), 0.0001f) * max(dot(Nmap, blueLightDir), 0.0), 0.0001f);
    
    float blueNdotL = max(dot(Nmap, blueLightDir), 0.0);
    vec3 blueLo = (kD * albedo / PI + blueSpecular) * blueRadiance * blueNdotL;

    vec3 ambient = vec3(0.04f) * albedo;

    vec3 color = ambient + Lo + blueLo;

    outColor = vec4(color, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(N, L),0.0,1.0)) + vec3(pow(clamp(dot(N, H),0.0,1.0), 160.0)) + ambient, 1.0);
    //outColor = vec4(albedo * vec3(clamp(dot(Nmap, L),0.0,1.0)) + vec3(pow(clamp(dot(Nmap, H),0.0,1.0), 160.0)) + ambient, 1.0);
	//outColor = vec4((Nmap+1.0f)*0.5f, 1.0);
	//outColor = vec4(texture(normalMap, fragUV).xyz, 1.0);
}
