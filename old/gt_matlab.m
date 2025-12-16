% MATLAB script to compute ground truth for 2-pool Bloch-McConnell equations
% This provides reference solutions to validate the PINN

clear; close all; clc;

%% Parameters for test case
% Relaxation times
T1w = 1.5;      % Water T1 (seconds)
T2w = 0.05;     % Water T2 (seconds)
T1s = 1.0;      % Solute T1 (seconds)
T2s = 0.01;     % Solute T2 (seconds)

% Exchange rates
k_ws = 100;     % Water -> Solute exchange rate (Hz)
k_sw = 100;    % Solute -> Water exchange rate (Hz)

% Offset frequencies
delta_omega_w = 0;      % Water offset (Hz) - on resonance
delta_omega_s = 200;    % Solute offset (Hz) - 200 Hz off-resonance

% RF field
omega1 = 100;   % RF field strength (Hz)

% Equilibrium magnetizations
M_w0 = 1.0;     % Water equilibrium magnetization
M_s0 = 0.01;    % Solute equilibrium magnetization (1% pool)

% Time settings
t_max = 5.0;    % Maximum saturation time (seconds)
t_span = [0, t_max];
t_eval = linspace(0, t_max, 501);  % Time points for evaluation

%% Initial conditions
% At thermal equilibrium: transverse magnetization = 0, longitudinal = M0
% Order: [M_wx, M_wy, M_wz, M_sx, M_sy, M_sz]
M0 = [0; 0; M_w0; 0; 0; M_s0];

%% Define the ODE system
% Bloch-McConnell equations for 2-pool system
bloch_ode = @(t, M) bloch_mcconnell_2pool(t, M, ...
    k_ws, k_sw, delta_omega_w, delta_omega_s, omega1, ...
    T1w, T2w, T1s, T2s, M_w0, M_s0);

%% Solve the ODE system
fprintf('Solving 2-pool Bloch-McConnell equations...\n');
fprintf('Parameters:\n');
fprintf('  k_ws = %.1f Hz, k_sw = %.1f Hz\n', k_ws, k_sw);
fprintf('  delta_omega_s = %.1f Hz, omega1 = %.1f Hz\n', delta_omega_s, omega1);
fprintf('  T1w = %.2f s, T2w = %.3f s\n', T1w, T2w);
fprintf('  T1s = %.2f s, T2s = %.3f s\n', T1s, T2s);
fprintf('  M_w0 = %.2f, M_s0 = %.3f\n\n', M_w0, M_s0);

% Use ode15s for stiff ODEs (Bloch-McConnell can be stiff)
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t_sol, M_sol] = ode15s(bloch_ode, t_eval, M0, options);

fprintf('ODE solution computed successfully!\n');
fprintf('Solution size: %d time points\n\n', length(t_sol));

%% Extract magnetization components
M_wx = M_sol(:, 1);
M_wy = M_sol(:, 2);
M_wz = M_sol(:, 3);
M_sx = M_sol(:, 4);
M_sy = M_sol(:, 5);
M_sz = M_sol(:, 6);

%% Save results to file for Python validation
results = struct();
results.t = t_sol;
results.M_wx = M_wx;
results.M_wy = M_wy;
results.M_wz = M_wz;
results.M_sx = M_sx;
results.M_sy = M_sy;
results.M_sz = M_sz;
results.parameters = struct('k_ws', k_ws, 'k_sw', k_sw, ...
    'delta_omega_w', delta_omega_w, 'delta_omega_s', delta_omega_s, ...
    'omega1', omega1, 'T1w', T1w, 'T2w', T2w, 'T1s', T1s, 'T2s', T2s, ...
    'M_w0', M_w0, 'M_s0', M_s0);

save('bloch_ground_truth.mat', '-struct', 'results');
fprintf('Results saved to: bloch_ground_truth.mat\n\n');

% Also save as CSV for easy Python loading
csvwrite('bloch_ground_truth_t.csv', t_sol);
csvwrite('bloch_ground_truth_M.csv', M_sol);
fprintf('Results also saved to CSV files\n\n');

%% Plot results
figure('Position', [100, 100, 1200, 800]);

% Water pool magnetization
subplot(2, 3, 1);
plot(t_sol, M_wx, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{wx}');
title('Water Pool - Transverse X');
grid on;

subplot(2, 3, 2);
plot(t_sol, M_wy, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{wy}');
title('Water Pool - Transverse Y');
grid on;

subplot(2, 3, 3);
plot(t_sol, M_wz, 'b-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{wz}');
title('Water Pool - Longitudinal Z');
grid on;
hold on;
plot([t_sol(1), t_sol(end)], [M_w0, M_w0], 'k--', 'LineWidth', 1);
legend('M_{wz}', 'M_{w0}');

% Solute pool magnetization
subplot(2, 3, 4);
plot(t_sol, M_sx, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{sx}');
title('Solute Pool - Transverse X');
grid on;

subplot(2, 3, 5);
plot(t_sol, M_sy, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{sy}');
title('Solute Pool - Transverse Y');
grid on;

subplot(2, 3, 6);
plot(t_sol, M_sz, 'r-', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('M_{sz}');
title('Solute Pool - Longitudinal Z');
grid on;
hold on;
plot([t_sol(1), t_sol(end)], [M_s0, M_s0], 'k--', 'LineWidth', 1);
legend('M_{sz}', 'M_{s0}');

sgtitle('2-Pool Bloch-McConnell Ground Truth Solution');
saveas(gcf, 'bloch_ground_truth.png');
fprintf('Plot saved to: bloch_ground_truth.png\n\n');

%% Compute Z-spectrum (saturation vs offset frequency)
fprintf('Computing Z-spectrum...\n');
offsets = linspace(-1000, 1000, 51);  % Offset frequencies (Hz)
t_sat = 3.0;  % Saturation time (seconds)
z_spectrum = zeros(size(offsets));

for i = 1:length(offsets)
    delta_omega_s_sweep = offsets(i);
    
    % Define ODE with current offset
    bloch_ode_sweep = @(t, M) bloch_mcconnell_2pool(t, M, ...
        k_ws, k_sw, delta_omega_w, delta_omega_s_sweep, omega1, ...
        T1w, T2w, T1s, T2s, M_w0, M_s0);
    
    % Solve to saturation time
    [~, M_sweep] = ode15s(bloch_ode_sweep, [0, t_sat], M0, options);
    
    % Z-spectrum is normalized water longitudinal magnetization
    z_spectrum(i) = M_sweep(end, 3) / M_w0;
end

fprintf('Z-spectrum computed!\n\n');

% Plot Z-spectrum
figure('Position', [100, 100, 800, 500]);
plot(offsets, z_spectrum, 'b-', 'LineWidth', 2);
xlabel('Offset Frequency (Hz)');
ylabel('M_{wz}/M_{w0}');
title(sprintf('Z-Spectrum (t_{sat} = %.1f s)', t_sat));
grid on;
xlim([min(offsets), max(offsets)]);
ylim([0, 1.1]);

% Mark the solute pool offset
hold on;
plot(delta_omega_s, interp1(offsets, z_spectrum, delta_omega_s), ...
    'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('Z-spectrum', 'Solute offset', 'Location', 'best');

saveas(gcf, 'z_spectrum.png');
fprintf('Z-spectrum plot saved to: z_spectrum.png\n\n');

% Save Z-spectrum data
z_spectrum_data = [offsets', z_spectrum'];
csvwrite('z_spectrum.csv', z_spectrum_data);
fprintf('Z-spectrum data saved to: z_spectrum.csv\n');

fprintf('All computations complete!\n');

%% ODE function definition
function dMdt = bloch_mcconnell_2pool(t, M, k_ws, k_sw, ...
    delta_omega_w, delta_omega_s, omega1, ...
    T1w, T2w, T1s, T2s, M_w0, M_s0)
    % 2-pool Bloch-McConnell equations
    % M = [M_wx, M_wy, M_wz, M_sx, M_sy, M_sz]'
    
    % Extract components
    M_wx = M(1);
    M_wy = M(2);
    M_wz = M(3);
    M_sx = M(4);
    M_sy = M(5);
    M_sz = M(6);
    
    % Initialize derivative vector
    dMdt = zeros(6, 1);
    
    % Water pool equations
    dMdt(1) = -k_ws * M_wx + delta_omega_w * M_wy + k_sw * M_sx - M_wx / T2w;
    dMdt(2) = delta_omega_w * M_wx - k_ws * M_wy - omega1 * M_wz - M_wy / T2w;
    dMdt(3) = omega1 * M_wy - k_ws * M_wz + k_sw * M_sz + (M_w0 - M_wz) / T1w;
    
    % Solute pool equations
    dMdt(4) = k_ws * M_wx - k_sw * M_sx + delta_omega_s * M_sy - M_sx / T2s;
    dMdt(5) = k_ws * M_wy + delta_omega_s * M_sx - k_sw * M_sy - omega1 * M_sz - M_sy / T2s;
    dMdt(6) = k_ws * M_wz + omega1 * M_sy - k_sw * M_sz + (M_s0 - M_sz) / T1s;
end