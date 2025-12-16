function bm_longitudinal_2pool_GT
    % Simple, stable longitudinal 2-pool GT

    T1a = 3.5;  R1a = 1/T1a;
    T1b = 3.0;  R1b = 1/T1b;

    k_ab = 1.0;   % a -> b
    k_ba = 1.0;   % b -> a

    M0a = 1.0;
    M0b = 0.0;

    tspan = [0 5];   % seconds

    M0 = [M0a; M0b];   % [Mza(0); Mzb(0)]

    params.R1a = R1a;
    params.R1b = R1b;
    params.k_ab = k_ab;
    params.k_ba = k_ba;
    params.M0a = M0a;
    params.M0b = M0b;

    ode_fun = @(t,M) rhs_longitudinal(t,M,params);
    opts = odeset('RelTol',1e-10,'AbsTol',1e-12);
    [t, M] = ode45(ode_fun, tspan, M0, opts);

    % M(:,1) = Mza, M(:,2) = Mzb
    figure; plot(t, M(:,1), 'b-', 'LineWidth', 2);
    hold on; plot(t, M(:,2), 'r-', 'LineWidth', 2);
    xlabel('t [s]'); ylabel('M_{za}');
    legend('M_{za}', 'M_{zb}');
    title('Longitudinal 2-pool: water M_z');

    GT_long.t = t;
    GT_long.M = M;
    save(fullfile(pwd, 'pinns-mri/longitudinal_2pool/gt', 'bm_longitudinal_2pool_GT.mat'), 'GT_long');
end

function dMdt = rhs_longitudinal(~, M, p)
    Mza = M(1); Mzb = M(2);
    R1a = p.R1a; R1b = p.R1b;
    k_ab = p.k_ab; k_ba = p.k_ba;
    M0a = p.M0a; M0b = p.M0b;

    dMza = -R1a*(Mza - M0a) - k_ab*Mza + k_ba*Mzb;
    dMzb = -R1b*(Mzb - M0b) - k_ba*Mzb + k_ab*Mza;

    dMdt = [dMza; dMzb];
end
