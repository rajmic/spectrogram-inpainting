function [ x_hat, objective, times ] = proxgrad(prox, grad, L, startpoint, ...
    iterations, fun, varargin)
% PROXGRAD Proximal gradient algorithm. It can be used as the projected
% gradient algorithm when the input proximal operator is a projection, or
% as FISTA if the proximal operator is thresholding.
%
% The problem to solve is
%
%      x_hat = arg min g(x) + h(x)
%
% where we know the proximal operator of h and the gradient of g.
%
% The basic algorithm and the FISTA-like acceleration are taken from
% Beck, A. & Teboulle, M. A Fast Iterative Shrinkage-Thresholding Algorithm
% for Linear Inverse Problems SIAM Journal on Imaging Sciences, SIAM, 2009,
% 2, 183-202.
%
% The Nesterov acceleration is taken from
% Ryan Tibshirani, Proximal Gradient Descent and Acceleration, p. 22
%
% Input parameters
%       prox           proximal operator of c*h(x) used as prox(x, c) where
%                      c > 0 is a constant
%       grad           gradient of g(x), assumed to be linear
%       L              operator norm of the gradient
%       startpoint     starting point of the algorithm
%       iterations     number of iterations of the algorithm
%       fun            objective function
%
% Output parameters
%       x_hat          solution
%       objective      values of the objective function during iterations
%       times          cumulative computation time during iterations
%
% Date: 24/11/2023
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@mensa.cz

pars = inputParser;
pars.KeepUnmatched = true;
addParameter(pars, "saveall", false)
addParameter(pars, "acceleration", "")
parse(pars, varargin{:})
saveall = pars.Results.saveall;
acceleration = pars.Results.acceleration;

% initialization
x_new = startpoint;
y = startpoint;
t_new = 1;

if nargout > 1
    objective = NaN(iterations, 1);
end

if nargout > 2
    times = NaN(iterations, 1);
    tic
end

if saveall
    x_hat = NaN(length(startpoint), iterations);
end

% iterations
for i = 1:iterations
    
    % save the x and t from the previous iteration
    x_old = x_new;
    t_old = t_new;
    
    % update the main variable
    x_new = prox(y - (1/L)*grad(y), 1/L);
    
    % acceleration
    if strcmpi(acceleration, "fista")

        % update the extrapolation step size
        t_new = (1 + sqrt(1 + 4*t_old^2))/2;
        
        % update the auxiliary variable
        y = x_new + (t_old - 1)/t_new * (x_new - x_old);

    elseif strcmpi(acceleration, "nesterov")
    
        y = x_new + (i-2)/(i+1)*(x_new-x_old);
    
    else

        y = x_new;

    end

    % saving the objective
    if nargout > 1
        objective(i) = fun(x_new);
    end
    
    % saving the time
    if nargout > 2
        times(i) = toc;
    end
    
    % update the solution
    if saveall
        x_hat(:,i) = x_new;
    else
        x_hat = x_new;
    end
end

end

