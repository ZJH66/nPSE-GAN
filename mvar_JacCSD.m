function [J_x J_A] = mvar_JacCSD(A,x_ext,p)
%Complex Step Differentiation (CSD) for computing the Jacobian matrix 
%
% Arguments and variables:
% CH: number of channels
% p: MVAR model order
% x_ext: (CH*p x 1) Input vector of the MVAR model, x_ext(k-1) = [x(k-1); x(k-2); ...; x(k-p)]
% A: (CH*p x CH*p) All MVAR matrices in a vector column, A = [A1 A2 ... Ap; I 0 0 .. 0; 0 I 0 ... 0; ...; 0 0 ... I 0]
% F: (CH*p x 1) Output of the model
% x_ext(k) = F(x_ext(k-1)) = A * x_ext(k-1)
%
% J_x: (CH*p x CH*p) Jacobian matrix through Complex Step Differentiation (CSD) --> df/dx
% J_A: (CH*p x CH*CH*p) Jacobian matrix through Complex Step Differentiation (CSD) --> df/dA
%
% f(x0 + ih) = f(x0) + ih*f'(x0) + (ih)^2*f''(x0)/2! + ...
% --> f'(x0) = imag( f(x0+ih)/h ) (Approximately)
% 
% 
% Reference:
% [1] M. S. Ridout, “Statistical Applications of the Complex-Step Method of Numerical Differentiation,? 
% The American Statistician, vol. 63, no. 1, pp. 66-74, 2009.
% 
% 
% See also: 'Complex step Jacobian' written by Yi Cao, available at:
% http://www.mathworks.com/matlabcentral/fileexchange/18176-complex-step-jacobian
%  
% Written by: Amir Omidvarnia

CH = length(x_ext)/p;         % Number of states or channels (CH*p)
[CHp CHp] = size(A);          % MVAR coefficients matrix (CH*p x CH*p)
h = .00000001;
J_x = zeros(CHp,CHp);         % df/dx
J_A = zeros(CHp,CH*CHp);

%% df/dx
for i = 1 : CHp
    tmp_x = x_ext;
    tmp_x(i) = tmp_x(i) + 1i*h;            % x0 + ih
    tmp_fx = A * tmp_x + randn(CHp,1);     % f(x0+ih)
    J_x(:,i) = imag(tmp_fx/h);             % f'(x0) = imag( f(x0+ih)/h ) (Approximately) ----> df/dx
end

%% df/dA ----> DfDa = -J_A 
a = reshape(A(1:CH,:)',CH*CHp,1);
for i = 1 : CH*CHp
    tmp_a = a;
    tmp_a(i) = tmp_a(i) + 1i*h;            % a0 + ih
    tmp_A = reshape(tmp_a, CH*p, CH)';
    for r = 2 : p
        tmp_A((r-1)*CH+1:r*CH,(r-2)*CH+1:(r-1)*CH) = eye(CH);
    end
    tmp_fa = tmp_A * x_ext + randn(CHp,1); % f(a0+ih)
    J_A(:,i) = imag(tmp_fa/h);
end

