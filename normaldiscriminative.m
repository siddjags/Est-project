clc;
clear;
close all;

d_matrix = readmatrix('/Users/siddhantjagtap/Documents/CMU Assignments/752/Project/Air quality final/regression.csv');

feature_matrix = d_matrix(:,2:9);
label_column = d_matrix(:,end);


subplot(4,2,1);
scatter(feature_matrix(:,1),label_column);
xlabel('bp');
ylabel('uvpm');

subplot(4,2,2);
scatter(feature_matrix(:,2),label_column);
xlabel('co');
ylabel('uvpm');


subplot(4,2,3);
scatter(feature_matrix(:,3),label_column);
xlabel('no');
ylabel('uvpm');

subplot(4,2,4);
scatter(feature_matrix(:,4),label_column);
xlabel('no2');
ylabel('uvpm');

subplot(4,2,5);
scatter(feature_matrix(:,5),label_column);
xlabel('nox');
ylabel('uvpm');

subplot(4,2,6);
scatter(feature_matrix(:,6),label_column);
xlabel('relative_humidity');
ylabel('uvpm');

subplot(4,2,7);
scatter(feature_matrix(:,7),label_column);
xlabel('temperature');
ylabel('uvpm');

subplot(4,2,8);
scatter(feature_matrix(:,8),label_column);
xlabel('PM2.5');
ylabel('uvpm');

% From above plots, we can conclude that the uvpm values are dependent on
% each of the 4 pollutant concentrations.

% Split into train,test,validation:

n_samples = size(d_matrix,1); % Total number of points in dataset

prop_train = ceil(0.6*n_samples);
prop_valid = ceil(0.2*n_samples);
prop_test = n_samples - (prop_valid + prop_train);


% Feature Matrices:

train_features = feature_matrix(1:prop_train,:);
valid_features = feature_matrix((1+prop_train):(prop_train+prop_valid),:);
test_features = feature_matrix((prop_train+prop_valid+1):n_samples,:);

% Label Matrices:

train_labels = label_column(1:prop_train,:);
valid_labels = label_column((1+prop_train):(prop_train+prop_valid),:);
test_labels = label_column((prop_train+prop_valid+1):n_samples,:);

% We try to apply the Normal Discriminative Model to the data using 4 features:

train_input = [ones(size(train_features,1),1), train_features(:,2:8)].^1;
valid_input = [ones(size(valid_features,1),1), valid_features(:,2:8)].^1;
test_input = [ones(size(test_features,1),1), test_features(:,2:8)].^1;

train_input = [ones(size(train_features,1),1), train_features(:,2:8).^1, train_features(:,2:8).^2, train_features(:,2:8).^3];
valid_input = [ones(size(valid_features,1),1), valid_features(:,2:8).^1, valid_features(:,2:8).^2, valid_features(:,2:8).^3];
test_input = [ones(size(test_features,1),1), test_features(:,2:8).^1, test_features(:,2:8).^2, test_features(:,2:8).^3];

% Training the model:

lambda = 0;

theta_mle = (inv(train_input'*train_input + lambda*eye(size(train_input,2))))*train_input'*train_labels;

% Predicting on training set:

train_predict = train_input*theta_mle;

train_mse = (1/size(train_labels,1))*(sum((train_labels - train_predict).^2));

train_normal_mse = train_mse/var(train_labels)

% Predicting on validation set:

valid_predict = valid_input*theta_mle;

valid_mse = (1/size(valid_labels,1))*(sum((valid_labels - valid_predict).^2));

valid_normal_mse = valid_mse/var(valid_labels)

