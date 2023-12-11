function [ u ] = pd_controller(~, s, s_des, params)
%PD_CONTROLLER  PD controller for the height
%
%   s: 2x1 vector containing the current state [z; v_z]，当前状态，高度和垂直速度
%   s_des: 2x1 vector containing desired state [z; v_z]，目标状态，高度和垂直速度
%   params: robot parameters

u = 0;
kp=100;kd=15;
zdes=0
%u=m(z¨d​es+Kp​e+Kv​e˙+g)
u = params.mass*(zdes+sum([kp;kd].*(s_des-s))+params.gravity);
if u>params.u_max
    u=params.u_max;
end


% FILL IN YOUR CODE HERE


end

