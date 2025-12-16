function bloch_mcconnell_2pool_GT
    % Parameters (match Python)
    T1w = 1.5;   % s
    T2w = 0.1;   % s
    T1s = 1.0;   % s
    T2s = 0.05;  % s

    k_ws = 5.0;          % water -> solute
    k_sw = 5.0;          % solute -> water
    dw_w = 0.0;          % water offset
    dw_s = 100.0;        % solute offset
    omega1 = 50.0;       % RF amplitude

    M_w0 = 1.0;          % water equilibrium Mz
    M_s0 = 0.0;          % solute equilibrium Mz

    t_max = 5.0;         % s
    tspan = [0 t_max];

    % Initial condition: (M_wx, M_wy, M_wz, M_sx, M_sy, M_sz)
    M0 = [0; 0; M_w0; 0; 0; M_s0];

    % Pack parameters
    params.T1w = T1w;
    params.T2w = T2w;
    params.T1s = T1s;
    params.T2s = T2s;
    params.k_ws = k_ws;
    params.k_sw = k_sw;
    params.dw_w = dw_w;
    params.dw_s = dw_s;
    params.omega1 = omega1;
    params.M_w0 = M_w0;
    params.M_s0 = M_s0;

    % Solve ODE
    ode_fun = @(t, M) bloch_mcconnell_rhs(t, M, params);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
    [t, M] = ode45(ode_fun, tspan, M0, opts);

    % M columns: [M_wx, M_wy, M_wz, M_sx, M_sy, M_sz]
    M_wx = M(:,1); M_wy = M(:,2); M_wz = M(:,3);
    M_sx = M(:,4); M_sy = M(:,5); M_sz = M(:,6);

    % Example: plot water Mz
    figure; plot(t, M_wz, 'b-', 'LineWidth', 2);
    xlabel('t [s]'); ylabel('M_{wz}');
    title('Water M_z (GT)');

    % Save GT to file for Python
    GT.t   = t;
    GT.M   = M;   % 6 columns
    save(fullfile(pwd, 'pinns-mri/output_gt', 'bloch_mcconnell_2pool_GT.mat'), 'GT');
end


function dMdt = bloch_mcconnell_rhs(~, M, p)
    % Unpack state
    M_wx = M(1); M_wy = M(2); M_wz = M(3);
    M_sx = M(4); M_sy = M(5); M_sz = M(6);

    % Unpack parameters
    T1w = p.T1w; T2w = p.T2w;
    T1s = p.T1s; T2s = p.T2s;
    k_ws = p.k_ws;   % water -> solute
    k_sw = p.k_sw;   % solute -> water
    dw_w = p.dw_w;
    dw_s = p.dw_s;
    omega1 = p.omega1;
    M_w0 = p.M_w0;
    M_s0 = p.M_s0;

    % Water pool (matches Python residuals)
    dM_wx = -k_ws * M_wx + dw_w * M_wy + k_sw * M_sx - M_wx / T2w;
    dM_wy =  dw_w * M_wx - k_ws * M_wy - omega1 * M_wz - M_wy / T2w;
    dM_wz =  omega1 * M_wy - k_ws * M_wz + k_sw * M_sz + (M_w0 - M_wz) / T1w;

    % Solute pool
    dM_sx = -k_sw * M_sx + dw_s * M_sy + k_ws * M_wx - M_sx / T2s;
    dM_sy =  dw_s * M_sx - k_sw * M_sy - omega1 * M_sz - M_sy / T2s;
    dM_sz =  omega1 * M_sy - k_sw * M_sz + k_ws * M_wz + (M_s0 - M_sz) / T1s;

    dMdt = [dM_wx; dM_wy; dM_wz; dM_sx; dM_sy; dM_sz];
end
