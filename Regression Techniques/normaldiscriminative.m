clc;
clear;
close all;

d_matrix = readmatrix('/Users/siddhantjagtap/Documents/CMU Assignments/752/Project/Air quality final/regression.csv');

feature_matrix = [d_matrix(:,1:3) d_matrix(:,5:9)];
label_column = d_matrix(:,4);


subplot(4,2,1);
scatter(feature_matrix(:,1),label_column);
xlabel('bp');
ylabel('NO2');

subplot(4,2,2);
scatter(feature_matrix(:,2),label_column);
xlabel('co');
ylabel('NO2');


subplot(4,2,3);
scatter(feature_matrix(:,3),label_column);
xlabel('no');
ylabel('NO2');
% 
subplot(4,2,4);
scatter(feature_matrix(:,4),label_column);
xlabel('NOx');
ylabel('NO2');
% 
subplot(4,2,5);
scatter(feature_matrix(:,5),label_column);
xlabel('relative humidity');
ylabel('No2');
% 
subplot(4,2,6);
scatter(feature_matrix(:,6),label_column);
xlabel('relative_temp');
ylabel('NO2');
% 
subplot(4,2,7);
scatter(feature_matrix(:,7),label_column);
xlabel('pm2.5');
ylabel('NO2');

subplot(4,2,8);
scatter(feature_matrix(:,8),label_column);
xlabel('uvpm');
ylabel('NO2');
