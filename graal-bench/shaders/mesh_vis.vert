#version 450
#include "encode_normal.glsl"

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec3 a_tangent;

layout(location=0) out vec3 v_position_vs;
layout(location=1) out vec3 v_normal_vs;
layout(location=2) out vec3 v_tangent_vs;

layout(set=0,binding=0,std140) uniform Globals {
    mat4 u_view_matrix;
    mat4 u_proj_matrix;
    mat4 u_view_proj_matrix;
    mat4 u_inverse_proj_matrix;
};

layout(set=2,binding=0,std140) uniform PerObject {
    mat4 u_model_matrix;
    mat4 u_model_it_matrix;
};

void main() {
    vec4 pos = vec4(a_position,1.0f);
    vec4 n = vec4(a_normal, 0.0);
    vec4 t = vec4(a_tangent, 0.0);

    mat4 model_view_matrix = u_view_matrix * u_model_matrix;

    gl_Position = u_view_proj_matrix * u_model_matrix * pos;

    v_position_vs = (model_view_matrix * pos).xyz;
    v_normal_vs = (u_view_matrix * u_model_it_matrix * n).xyz;
    v_tangent_vs = (model_view_matrix * t).xyz;
}
