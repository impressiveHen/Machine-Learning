%% 
clc;
clear all;
% hw1-4-b
% draw random points from two normal distribution Gaussian 
alpha_case = [10,2];
variance_case = [2,10];

% condition A
alpha = alpha_case(1);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(1);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);
figure;
subplot(1,2,1);
% 'ro' red circle
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
% 'bo' blue circle
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
hold off;
axis([-13 13 -10 10])
   
% condition B
alpha = alpha_case(2);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(2);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
% mvnrnd: Multivariate normal random numbers
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);
subplot(1,2,2);
% 'ro': red circle
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
hold off;
% set axis limit
axis([-13 13 -9 9])

%%
% hw1-4-c
% draw the arrow of the PCA max eigen vector

% drawArrow(x,y) -> draws arrow from (x(1),y(1)) to (x(2),y(2)), 'k' for
% black 
drawArrow = @(x,y) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),'k' );

% condition A
alpha = alpha_case(1);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(1);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);

all_data = [x1;x2];
mean_all_data = mean(all_data);
cov_mat = cov(all_data);
[uSVD,sSVD,vSVD] = svd(cov_mat);
figure;
subplot(1,2,1);
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
% data dimension 2 -> covariance matrix 2*2 -> eigenvector of covariance
% matrix 1*2
maxEig = uSVD(:,1);
% draw projection axis arrow from the mean of the data, to a point pluss
% eigenvector 
drawArrow([mean_all_data(1),mean_all_data(1) + maxEig(1)*5],...
    [mean_all_data(1),mean_all_data(1)+maxEig(2)*5]);
hold off;

% condition B
alpha = alpha_case(2);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(2);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);

all_data = [x1;x2];
cov_mat = cov(all_data);
[uSVD,sSVD,vSVD] = svd(cov_mat);
subplot(1,2,2);
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
maxEig = uSVD(:,1);
% draw the principle axis, the line produce by the 
drawArrow([mean_all_data(1),mean_all_data(1) + maxEig(1)*5],...
    [mean_all_data(1),mean_all_data(1)+maxEig(2)*5]);
hold off;

%% 
% hw1-4-d
% in LDA z = w' * x -> the projection that best separates the classes 
% w* = (sigma0 + sigma1)^-1 * (u1 - u0) 

% condition A
alpha = alpha_case(1);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(1);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);

w_star = inv(sigma1 + sigma2)' *  (u2 - u1);

subplot(1,2,1);
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
% draw the principle axis, the projection axis from the mean of the  data
% to a point plus the w* vector of LDA
drawArrow([mean_all_data(1),mean_all_data(1) + w_star(1)*0.5],...
    [mean_all_data(1),mean_all_data(1)+w_star(2)*0.5]);
hold off;

% condition B
alpha = alpha_case(2);
u = [alpha;0];
u1 = u;
u2 = -u;
variance = variance_case(2);
sigma = [1,0;0,variance];
sigma1 = sigma;
sigma2 = sigma;
x1 = mvnrnd(u1,sigma1,1000);
x2 = mvnrnd(u2,sigma2,1000);

w_star = inv(sigma1 + sigma2)' *  (u2 - u1);

subplot(1,2,2);
plot(x1(:,1),x1(:,2),'ro', 'MarkerSize', 2);
hold on;
plot(x2(:,1),x2(:,2),'bo', 'MarkerSize', 2);
% draw the principle axis, the line produce by the 
drawArrow([mean_all_data(1),mean_all_data(1) + w_star(1)*2],...
    [mean_all_data(1),mean_all_data(1)+w_star(2)*2]);
hold off;




