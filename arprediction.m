clc;
clear;
close all;

d_matrix = readmatrix('/Users/siddhantjagtap/Documents/CMU Assignments/752/Project/Air quality final/timeseries.csv');

time_data = d_matrix(:,end);

n = 0:size(time_data,1)-1;

plot(time_data)

% Assume that this is an Auto-Regressive Process:
% Convert process to zero mean process:

n = ceil(0.9*size(time_data,1));

train_data = time_data(1:n,1);
test_data = time_data((n+1):size(d_matrix,1),1);

%plot(train_data)
%hold on;
%plot(test_data)

u_n = train_data - mean(train_data);

% We set tau = 3 and l = 5:

% Estimating ru(m):

l = 450;

ru = zeros(l,1);

for m = 0:(l-1)
    sum = 0;
    for i = m+1:size(train_data,1)
        sum = sum + u_n(i)*u_n(i-m);
    end
    ru(m+1,1) = sum/(size(train_data,1)-(m));
end

E_x = toeplitz(ru');

% We assume label is time-shifted version of input u_n:

rdu = zeros(l,1);

tau = 1;

for t = tau:(tau+(l-1))
    sum = 0;
    for i = t+1:size(train_data,1)
        sum = sum + u_n(i)*u_n(i-t);
    end
    rdu(t-(tau-1),1) = sum/(size(train_data,1)-(t));
end

p = rdu;

% Estimating theta_hat using normal equations:

w_hat = (inv(E_x))*p;

% Making future predictions:

init = size(train_data,1)+1;
pn = zeros(size(test_data,1),1);


%x_n = [train_data(init-tau,1);train_data(init-(tau+1),1);train_data(init-(tau+2),1);train_data(init-(tau+3),1);train_data(init-(tau+4),1)];
x_n = train_data(init-tau,1);

for i = 1:(l-1)
    x_n = [x_n;train_data(init-(tau+i),1)];
end

%for j = 1:l


%for i = 1:size(pn,1)
%    pn(i,1) = w_hat'*

for i = 1:size(test_data,1)
    pn(i,1) = w_hat'*x_n;
    concat = pn(i,1);
    x_n(1) = [];
    x_n = [x_n;concat];
end

plot([1:size(train_data,1)],train_data);
hold on;
plot([(size(train_data,1)+1):size(time_data,1)],pn);
hold on;
%plot([(size(train_data,1)+1):size(time_data,1)],test_data);


e_n = (test_data - pn).^2;

exp_sum = 0;

for i = 1:size(e_n)
    exp_sum = exp_sum + e_n(i);
end

expect = (1/size(test_data,1))*(exp_sum);

%mse = (1/size(test_data,1))*(sum((test_data - pn).^2));

normal_mse = expect/E_x(1,1);
    



