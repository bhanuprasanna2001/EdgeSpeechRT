function plot_metric_summary(csvPath)
%PLOT_METRIC_SUMMARY Plot PESQ/STOI/SI-SDR columns from a metric CSV.

t = readtable(csvPath);
figure;
tiledlayout(3, 1);
nexttile; bar(t.pesq); title('PESQ'); grid on;
nexttile; bar(t.stoi); title('STOI'); grid on;
nexttile; bar(t.si_sdr); title('SI-SDR'); grid on;
end
