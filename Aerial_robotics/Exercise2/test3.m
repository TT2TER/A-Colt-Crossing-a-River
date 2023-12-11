R=[[0.675, -0.1724, 0.7174];
              [0.2474, 0.9689,0];
              [ -0.6951,0.1775, 0.6967]]
wb=[0.2474; 0.9689;1]
wb=skew(wb)
dotR=inv(R')*wb
ws=dotR*R'
function S = skew(v)
a = size(v,1)==3;
n = size(v,round(1+a));
V = permute(v,[round(2-a),3,round(1+a)]);
I = repmat(eye(3),[1,1,n]);
S = cross(repmat(V,[1,3]),I);
end