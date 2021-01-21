clc;
clear all;
close all;

d_matrix = readmatrix('/Users/siddhantjagtap/Documents/CMU Assignments/752/Project/Air quality final/regression.csv');

feature_matrix = [d_matrix(:,2:3) d_matrix(:,5) d_matrix(:,9)];
label_column = d_matrix(:,4);

% Split dataset into 2 sets for cross-validation and training: 

n_samples = size(d_matrix,1);

prop_crossvalid = (ceil(0.8*n_samples)-4);
prop_test = n_samples - (prop_crossvalid);

% Feature Matrices:

cvalid_features = [ones(prop_crossvalid,1) feature_matrix(1:prop_crossvalid,:)];
test_features = feature_matrix((prop_crossvalid+1):size(feature_matrix,1),:);
test_features = [ones(size(test_features,1),1) test_features];

% Label Columns:

cvalid_labels = label_column(1:prop_crossvalid,:);
test_labels = label_column((1+prop_crossvalid):size(label_column,1),:);

% Performing 10 fold cross-validation for different L2 regularization penalties:

k = 10;
fold_size = size(cvalid_labels,1)/k;
lambda = 0:0.01:100;
D = eye(size(cvalid_features,2));
D(1,1)=0;
train_mse_list = zeros(1,size(lambda,2));
crossvalid_mse_list = zeros(1,size(lambda,2));

theta_mle_collection = [];

% Training using 9 folds each time:

for i = 1:size(lambda,2)
    % First Fold:
    
    train_f1 = cvalid_features(1:(9*fold_size),:);
    train_l1 = cvalid_labels(1:(9*fold_size),:);

    valid_f1 = cvalid_features((9*fold_size+1):(10*fold_size),:);
    valid_l1 = cvalid_labels((9*fold_size+1):(10*fold_size),:);

    theta_mle1 = (inv(train_f1'*train_f1 + lambda(1,i)*D))*train_f1'*train_l1;
    
    valid_p1 = valid_f1*theta_mle1;

    valid_mse_1 = (1/size(valid_l1,1))*(sum((valid_l1 - valid_p1).^2));
    
    train_mse_1 = (1/size(train_l1,1))*(sum((train_l1 - (train_f1*theta_mle1)).^2));

    % Second Fold:
    
    train_f2 = [cvalid_features(1:(8*fold_size),:);cvalid_features((9*fold_size +1):(10*fold_size),:)];
    train_l2 = [cvalid_labels(1:(8*fold_size),:);cvalid_labels((9*fold_size +1):(10*fold_size),:)];

    valid_f2 = cvalid_features((8*fold_size+1):(9*fold_size),:);
    valid_l2 = cvalid_labels((8*fold_size+1):(9*fold_size),:);
    
    theta_mle2 = (inv(train_f2'*train_f2 + lambda(1,i)*D))*train_f2'*train_l2;
    
    valid_p2 = valid_f2*theta_mle2;

    valid_mse_2 = (1/size(valid_l2,1))*(sum((valid_l2 - valid_p2).^2));
    
    train_mse_2 = (1/size(train_l2,1))*(sum((train_l2 - (train_f2*theta_mle2)).^2));
    
    % Third Fold:
    
    train_f3 = [cvalid_features(1:(7*fold_size),:);cvalid_features((8*fold_size +1):(10*fold_size),:)];
    train_l3 = [cvalid_labels(1:(7*fold_size),:);cvalid_labels((8*fold_size +1):(10*fold_size),:)];

    valid_f3 = cvalid_features((7*fold_size+1):(8*fold_size),:);
    valid_l3 = cvalid_labels((7*fold_size+1):(8*fold_size),:);
    
    theta_mle3 = (inv(train_f3'*train_f3 + lambda(1,i)*D))*train_f3'*train_l3;
    
    valid_p3 = valid_f3*theta_mle3;

    valid_mse_3 = (1/size(valid_l3,1))*(sum((valid_l3 - valid_p3).^2));
    
    train_mse_3 = (1/size(train_l3,1))*(sum((train_l3 - (train_f3*theta_mle3)).^2));
    
    % Fourth Fold:
    
    train_f4 = [cvalid_features(1:(6*fold_size),:);cvalid_features((7*fold_size +1):(10*fold_size),:)];
    train_l4 = [cvalid_labels(1:(6*fold_size),:);cvalid_labels((7*fold_size +1):(10*fold_size),:)];

    valid_f4 = cvalid_features((6*fold_size+1):(7*fold_size),:);
    valid_l4 = cvalid_labels((6*fold_size+1):(7*fold_size),:);
    
    theta_mle4 = (inv(train_f4'*train_f4 + lambda(1,i)*D))*train_f4'*train_l4;
    
    valid_p4 = valid_f4*theta_mle4;

    valid_mse_4 = (1/size(valid_l4,1))*(sum((valid_l4 - valid_p4).^2));
    
    train_mse_4 = (1/size(train_l4,1))*(sum((train_l4 - (train_f4*theta_mle4)).^2));
    
    % Fifth Fold:
    
    train_f5 = [cvalid_features(1:(5*fold_size),:);cvalid_features((6*fold_size +1):(10*fold_size),:)];
    train_l5 = [cvalid_labels(1:(5*fold_size),:);cvalid_labels((6*fold_size +1):(10*fold_size),:)];

    valid_f5 = cvalid_features((5*fold_size+1):(6*fold_size),:);
    valid_l5 = cvalid_labels((5*fold_size+1):(6*fold_size),:);
    
    theta_mle5 = (inv(train_f5'*train_f5 + lambda(1,i)*D))*train_f5'*train_l5;
    
    valid_p5 = valid_f5*theta_mle5;

    valid_mse_5 = (1/size(valid_l5,1))*(sum((valid_l5 - valid_p5).^2));
    
    train_mse_5 = (1/size(train_l5,1))*(sum((train_l5 - (train_f5*theta_mle5)).^2));
    
    % Sixth Fold:
    
    train_f6 = [cvalid_features(1:(4*fold_size),:);cvalid_features((5*fold_size +1):(10*fold_size),:)];
    train_l6 = [cvalid_labels(1:(4*fold_size),:);cvalid_labels((5*fold_size +1):(10*fold_size),:)];

    valid_f6 = cvalid_features((4*fold_size+1):(5*fold_size),:);
    valid_l6 = cvalid_labels((4*fold_size+1):(5*fold_size),:);
    
    theta_mle6 = (inv(train_f6'*train_f6 + lambda(1,i)*D))*train_f6'*train_l6;
    
    valid_p6 = valid_f6*theta_mle6;

    valid_mse_6 = (1/size(valid_l6,1))*(sum((valid_l6 - valid_p6).^2));
    
    train_mse_6 = (1/size(train_l6,1))*(sum((train_l6 - (train_f6*theta_mle6)).^2));
    
    % Seventh Fold:
    
    train_f7 = [cvalid_features(1:(3*fold_size),:);cvalid_features((4*fold_size +1):(10*fold_size),:)];
    train_l7 = [cvalid_labels(1:(3*fold_size),:);cvalid_labels((4*fold_size +1):(10*fold_size),:)];

    valid_f7 = cvalid_features((3*fold_size+1):(4*fold_size),:);
    valid_l7 = cvalid_labels((3*fold_size+1):(4*fold_size),:);
    
    theta_mle7 = (inv(train_f7'*train_f7 + lambda(1,i)*D))*train_f7'*train_l7;
    
    valid_p7 = valid_f7*theta_mle7;

    valid_mse_7 = (1/size(valid_l7,1))*(sum((valid_l7 - valid_p7).^2));
    
    train_mse_7 = (1/size(train_l7,1))*(sum((train_l7 - (train_f7*theta_mle7)).^2));
    
    % Eighth Fold:
    
    train_f8 = [cvalid_features(1:(2*fold_size),:);cvalid_features((3*fold_size +1):(10*fold_size),:)];
    train_l8 = [cvalid_labels(1:(2*fold_size),:);cvalid_labels((3*fold_size +1):(10*fold_size),:)];

    valid_f8 = cvalid_features((2*fold_size+1):(3*fold_size),:);
    valid_l8 = cvalid_labels((2*fold_size+1):(3*fold_size),:);
    
    theta_mle8 = (inv(train_f8'*train_f8 + lambda(1,i)*D))*train_f8'*train_l8;
    
    valid_p8 = valid_f8*theta_mle8;

    valid_mse_8 = (1/size(valid_l8,1))*(sum((valid_l8 - valid_p8).^2));
    
    train_mse_8 = (1/size(train_l8,1))*(sum((train_l8 - (train_f8*theta_mle8)).^2));
    
    % Ninth Fold:
    
    train_f9 = [cvalid_features(1:(1*fold_size),:);cvalid_features((2*fold_size +1):(10*fold_size),:)];
    train_l9 = [cvalid_labels(1:(1*fold_size),:);cvalid_labels((2*fold_size +1):(10*fold_size),:)];

    valid_f9 = cvalid_features((1*fold_size+1):(2*fold_size),:);
    valid_l9 = cvalid_labels((1*fold_size+1):(2*fold_size),:);
    
    theta_mle9 = (inv(train_f9'*train_f9 + lambda(1,i)*D))*train_f9'*train_l9;
    
    valid_p9 = valid_f9*theta_mle9;

    valid_mse_9 = (1/size(valid_l9,1))*(sum((valid_l9 - valid_p9).^2));
    
    train_mse_9 = (1/size(train_l9,1))*(sum((train_l9 - (train_f9*theta_mle9)).^2));
    
    % Last Fold:
    
    train_f10 = cvalid_features((fold_size+1):(10*fold_size),:);
    train_l10 = cvalid_labels((fold_size+1):(10*fold_size),:);

    valid_f10 = cvalid_features(1:(fold_size),:);
    valid_l10 = cvalid_labels(1:(fold_size),:);
    
    theta_mle10 = (inv(train_f10'*train_f10 + lambda(1,i)*D))*train_f10'*train_l10;
    
    valid_p10 = valid_f10*theta_mle10;

    valid_mse_10 = (1/size(valid_l10,1))*(sum((valid_l10 - valid_p10).^2));
    
    train_mse_10 = (1/size(train_l10,1))*(sum((train_l10 - (train_f10*theta_mle10)).^2));
    
    % Calculate avg. valid MSE:
    
    avg_valid_mse = (valid_mse_1 + valid_mse_2 + valid_mse_3 + valid_mse_4 + valid_mse_5 + valid_mse_6 + valid_mse_7 + valid_mse_8 + valid_mse_9 + valid_mse_10)/10;
    
    avg_train_mse = (train_mse_1 + train_mse_2 + train_mse_3 + train_mse_4 + train_mse_5 + train_mse_6 + train_mse_7 + train_mse_8 + train_mse_9 + train_mse_10)/10;
    
    theta_mle_avg = (0.1)*(theta_mle1 + theta_mle2 + theta_mle3 + theta_mle4 + theta_mle5 + theta_mle6 + theta_mle7 + theta_mle8 + theta_mle9 + theta_mle10);
    
    theta_mle_collection = [theta_mle_collection theta_mle_avg];
    
    %train_mse = (1/size(cvalid_features,1))*(sum((cvalid_labels - (cvalid_features*theta_mle_avg)).^2));
    
    train_mse_list(1,i) = avg_train_mse;
    crossvalid_mse_list(1,i) = avg_valid_mse;
end

 plot(lambda,train_mse_list);
 hold on;
 plot(lambda,crossvalid_mse_list);
 xlabel('lambda');
 ylabel('Empirical Risk');
 legend({'training risk', 'validation risk'},'Location','northwest');
 title('Empirical Risk vs Regularization Penalty');
    
 % From plot, select sweet-spot as lambda with lowest validation risk:

index = find(crossvalid_mse_list == min(crossvalid_mse_list));

theta_final = theta_mle_collection(:,index);

index_train = find(train_mse_list == min(train_mse_list));

% Making Predictions on Test dataset:

test_mse = (1/size(test_features,1))*(sum((test_labels - (test_features*theta_final)).^2));
    
normal_test_mse = (1/var(test_labels))*test_mse;

% Printing optimal lambda value:

fprintf('Optimal regularization penalty : %d \n', lambda(1,index)); 

% Printing testing risk:

fprintf('Normalized Test risk = %d \n',normal_test_mse);
