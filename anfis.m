df_list = {};   
for i = 3:95
    df = readmatrix(sprintf('C:/Users/Bora/Downloads/407/ebru_hoca_sepsis_dataset-2/ebru_hoca_dataset/sepsis/%d.csv', i));
    df = df(:, [1, 2, 4, 5, 8, 25]);
    df = rmmissing(df);
    df_list{end+1} = df;
end
df_all = vertcat(df_list{:});
df_all(df_all(:,5) == 0,:) = [];
df_all(df_all(:,4) < 8 | df_all(:,4) > 35,:) = [];
df_all(df_all(:,2) < 58 | df_all(:,2) > 165,:) = [];
df_all(df_all(:,3) < 44 | df_all(:,3) > 151,:) = [];
heart_rate = df_all(:, 1);
bp_systolic = df_all(:, 2);
resp = df_all(:, 3);
temp = df_all(:, 4);
wbc = df_all(:, 5);
qsofa = df_all(:, 6);
X = [heart_rate, bp_systolic, resp, temp, wbc];
y = qsofa;
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
X_train = X(cv.training, :);
y_train = y(cv.training);
X_test = X(cv.test, :);
y_test = y(cv.test);

% Define the ANFIS model
fis = anfis([X_train, y_train]);

% Training the model with anfis function
[fis, trnError, stepSize] = anfis([X_train, y_train],fis);

% Predictions on the test set
y_pred = evalfis(fis,X_test);       
rmse = sqrt(mean((y_pred - y_test).^2));
fprintf('RMSE: %.2f\n', rmse);