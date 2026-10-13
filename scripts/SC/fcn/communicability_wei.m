% Structural connectivity and communication-metric processing.
% Update the path variables below before running this script.
function   F = communicability_wei(CIJ)

%inputs
%           CIJ    weighted connection matrix
%           i      row
%           j      column
%
%outputs
%           F      communicability
%=================================================

N = size(CIJ,1);
B = sum(CIJ')';
C = diag(B);
D = C^(-(1/2));
E = D * CIJ * D;
F = expm(E);
F = F.*~eye(N);