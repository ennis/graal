#version 450

layout(location=0) in vec3 f_position_vs;
layout(location=1) in vec3 f_normal_vs;
layout(location=2) in vec3 f_tangent_vs;

layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_normal;
layout(location=2) out vec4 out_tangent;

layout(set=1,binding=0,std140) uniform Material {
    vec4 u_diffuse_color;
};

void main() {
    out_color = u_diffuse_color;
}
