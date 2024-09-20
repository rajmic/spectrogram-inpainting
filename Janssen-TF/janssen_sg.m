function [restored, objective, times, z] = janssen_sg(signal, sigprox, p, maxit, varargin)
% janssen_sg is an algorithm function based on [1], augmented by signal
% consistency governed by the proximal operator sigprox
%
% the subproblems of estimating the AR coefficients in and
% estimating the signal in are solved using the either by the
% Douglas-Rachford algorithm [2], its accelerated variant [3], or the
% Chambolle-Pock algorithm [4]
%
% [1] A. Janssen, R. Veldhuis and L. Vries, "Adaptive interpolation of
%     discrete-time signals that can be modeled as autoregressive
%     processes, " in IEEE Transactions on Acoustics, Speech, and Signal
%     Processing, vol. 34, no. 2, pp. 317-330, 1986, doi:
%     10.1109/TASSP.1986.1164824.
% [2] P. Combettes and J.-C. Pesquet, "Proximal Splitting Methods in Signal
%     Processing, " in Fixed-Point Algorithms for Inverse Problems in
%     Science and Engineering, vol. 49, pp. 185-212, 2009, doi:
%     10.1007/978-1-4419-9569-8_10.
% [3] I. Bayram, "Proximal Mappings Involving Almost Structured Matrices,"
%     in IEEE Signal Processing Letters, vol. 22, no. 12, pp. 2264-2268, 
%     2015, doi: 10.1109/LSP.2015.2476381.
% [4] A. Chambolle and T Pock, "A First-Order Primal-Dual Algorithm for
%     Convex Problems with Applications to Imaging," in Journal of
%     Mathematical Imaging and Vision, vol. 40, pp. 120–145, 2011,
%     doi: 10.1007/s10851-010-0251-1
%
% input arguments
%   signal        the input (degraded) signal
%   sigprox       proximal operator of the signal regularizer
%   p             order of the AR model
%   maxit         number of iterations of the whole Janssen algorithm
%   varargin      name-value pairs
%                 "algo" ("CP")      solver for the subproblem, accepted
%                                    choices are
%                                    "DR" ... Douglas-Rachford
%                                    "DRaccel" ... accelerated DR
%                                    "CP" ... Chambolle-Pock
%                                    "PG" ... proximal gradient
%                                    "ADMM" ... ADMM
%                 "algoit" (1000)    number of iterations for the
%                                    subsolver
%                 "K" ([])           linear operator if CP or ADMM is used
%                 "K_adj" ([])       adjoint operator if CP or ADMM is used
%                 "saveall" (false)  save the solution during iterations
%                 "mat" ("toeplitz") how to build the matrices needed for
%                                    the subproblems, accepted values are
%                                    "toeplitz", "xcorr", "conv", "fft"
%                 "decompose" (true) use the Cholesky decomposition to
%                                    substitute for the multiplication
%                                    with matrix inversion in the
%                                    subsolver
%                 "gamma" (10)       step length
%                 "verbose" (false)  print current iteration and other
%                                    otuputs
%                 "fun" ([])         regularization function
%
% output arguments
%   restored      the solution; if saveall is true, restored is of size
%                 length(signal) x maxit, or length(signal) x 1 otherwise
%   objective     values of the objective function during iterations
%   times         cumulative computation time during iterations
%
% Date: 09/05/2024
% By Ondrej Mokry
% Brno University of Technology
% Contact: ondrej.mokry@vut.cz

%% parse the inputs
% create the parser
pars = inputParser;
pars.KeepUnmatched = true;

% add optional name-value pairs
addParameter(pars, "algo", "CP")
addParameter(pars, "algoit", 1000)
addParameter(pars, "K", [])
addParameter(pars, "K_adj", [])
addParameter(pars, "saveall", false)
addParameter(pars, "mat", "toeplitz")
addParameter(pars, "decompose", true)
addParameter(pars, "gamma", 10)
addParameter(pars, "verbose", false)
addParameter(pars, "fun", [])

% parse
parse(pars, varargin{:})

% save the parsed results to nice variables
algo = pars.Results.algo;
algoit = pars.Results.algoit;
K = pars.Results.K;
K_adj = pars.Results.K_adj;
saveall = pars.Results.saveall;
mat = pars.Results.mat;
decompose = pars.Results.decompose;
gamma = pars.Results.gamma;
verbose = pars.Results.verbose;
fun = pars.Results.fun;

%% initialization
solution = signal;
N = length(signal);
if saveall
    restored = NaN(N, maxit);
end
z = [];

% manage fun
if isempty(fun)
    fun = @(x) 0;
end

% define some useful functions
if nargout > 1
    switch algo
        case {'CP', 'cp', 'ADMM', 'admm'}
            Q = @(x, c) 0.5*norm(fft(c, N+p).*fft(x, N+p))^2 / (N+p) + fun(K(x));
        otherwise
            Q = @(x, c) 0.5*norm(fft(c, N+p).*fft(x, N+p))^2 / (N+p) + fun(x);
    end
    objective = NaN(maxit, 1);
end

% if desired, start the timer
if nargout > 2
    times = NaN(maxit, 1);
    tic
end

%% main iteration
if verbose
    str = "";
end
for i = 1:maxit

    if verbose
        fprintf(repmat('\b', 1, strlength(str)))
        str = sprintf("iteration %d of %d", i, maxit);
        fprintf(str)
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                           AR model estimation                           %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    coef = lpc(solution, p)';
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                            signal estimation                            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % prepare the matrix AA
    if strcmpi(mat, "toeplitz")
        A  = toeplitz([coef; zeros(N-1, 1)], [coef(1), zeros(1, N-1)]);
        AA = A'*A;
        clear A
    elseif strcmpi(mat, "xcorr")
        b  = xcorr(coef, p)';
        AA = spdiags(ones(N, 1)*b, -p:p, N, N);
    elseif strcmpi(mat, "conv")
        b  = conv(coef, flip(coef))';
        AA = spdiags(ones(N, 1)*b, -p:p, N, N);
    elseif strcmpi(mat, "fft")
        b  = ifft(fft([coef' zeros(1, p)]) .* fft([flip(coef') zeros(1, p)]));
        AA = spdiags(ones(N, 1)*b, -p:p, N, N);
    end

    % check the positive definiteness using chol
    try
        dAA = decomposition(eye(N) + gamma*AA, "chol");
    catch
        if verbose
            warning("Stopped on decomposition in iteration %d.\n", i)
        end
        break
    end
    
    % check the conditioning
    if isIllConditioned(dAA)
        if verbose
            warning("Stopped on ill conditioning in iteration %d.\n", i)
        end
        break
    end

    switch algo
        case {"DRaccel", "draccel"}

            % solve the task
            solution = DouglasRachfordA(zeros(N+p, 1), coef, fun, sigprox, N, algoit, "alpha", gamma);

        case {"DR", "dr"}

            % set the parameters of the Douglas-Rachford algorithm
            DR.lambda = 1;
            DR.gamma = gamma;
            DR.y0 = solution;
            DR.maxit = algoit;
            DR.tol = -Inf;
    
            % set the parameters of the model
            DR.f = @(x) 0.5*norm(fft(coef, N+p).*fft(x, N+p))^2 / (N+p);
            if decompose
                DR.prox_f = @(x, t) dAA\x;
            else
                invmat = inv(eye(N) + gamma*AA);
                DR.prox_f = @(x, t) invmat*x; %#ok<MINV>
            end
            DR.g = fun;
            DR.prox_g = sigprox;
            DR.dim = N;
    
            % solve the task
            solution = DouglasRachford(DR, DR);

        case {"PG", "pg"}

            % compute the norm of A'*A
            normAA  = norm(AA);
            
            % save the crop function
            crop = @(vect, N) vect(1:N);
            
            % take the vector b from the matrix B = A'*A
            b  = AA(1:p+1, 1);
            Bcol = [b; zeros(N-p-1, 1); b(p+1:-1:2)];
            fftBcol = fft(Bcol);

            % the objective
            objfun = @(x) 0.5*norm(fft(coef, N+p).*fft(x, N+p))^2 / (N+p) + fun(x);

            % solve the task
            solution = proxgrad(...
                sigprox, @(x) crop(ifft(fftBcol.*fft([x; zeros(p, 1)])), N), ...
                normAA, solution, algoit, objfun, "acceleration", "FISTA");

        case {"CP", "cp"}

            % set the parameters of the Chambolle-Pock algorithm
            CP.theta = 1;
            CP.sigma = 1/gamma;
            CP.tau = gamma;
            CP.x0 = solution;
            CP.maxit = algoit;
            CP.tol = -Inf;
            CP.K = K;
            CP.K_adj = K_adj;
    
            % set the parameters of the model
            CP.g = @(x) 0.5*norm(fft(coef, N+p).*fft(x, N+p))^2 / (N+p);
            if decompose
                CP.prox_g = @(x, t) dAA\x;
            else
                invmat = inv(eye(N) + gamma*AA);
                CP.prox_g = @(x, t) invmat*x; %#ok<MINV>
            end
            CP.f = fun;
            CP.prox_f = sigprox;
            CP.dim = N;
    
            % solve the task
            solution = ChambollePock(CP, CP);

        case {"ADMM", "admm"}

            rho = 1/gamma;
            x = solution;
            z = K(x);
            u = zeros(size(z));
            prox_f = sigprox;
            if decompose
                prox_g = @(x, t) dAA\x;
            else
                invmat = inv(eye(N) + gamma*AA);
                prox_g = @(x, t) invmat*x; %#ok<MINV>
            end
            for it = 1:algoit
                x = prox_g(K_adj(z - u), 1/rho);
                z = prox_f(K(x) + u, 1/rho);
                u = u + K(x) - z;
            end
            solution = x;

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                       solution and objective update                     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % update the solution
    if saveall
        restored(:, i) = solution;
    else
        restored = solution;
    end
    
	% compute the objective value
    if nargout > 1
        objective(i) = Q(solution, coef);
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                                time update                              %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargout > 2
        times(i) = toc;
    end

end

if verbose
    fprintf(repmat('\b', 1, strlength(str)))
end

end