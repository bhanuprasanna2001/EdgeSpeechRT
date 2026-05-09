function report = compare_python_cpp_outputs(pythonTxt, cppTxt)
%COMPARE_PYTHON_CPP_OUTPUTS Compare float text dumps from Python and C++.

py = load(pythonTxt);
cpp = load(cppTxt);
n = min(numel(py), numel(cpp));
err = py(1:n) - cpp(1:n);
report.max_abs = max(abs(err));
report.rmse = sqrt(mean(err.^2));
report.samples_compared = n;
disp(report);
end
